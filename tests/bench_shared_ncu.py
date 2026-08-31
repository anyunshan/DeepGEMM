"""Minimal single-launch driver for profiling the SM90 shared-expert kernel.

Mirrors the --local-rank-idx / --ncu-profile-only interface the SM100 ncu script
uses, so ncu can attach one process per rank and replay a single kernel launch.

The profiled rank does exactly one launch; peers loop so all 8 ranks stay alive
through the profiled rank's replays (the kernel has in-kernel NVLink barriers, so
a peer that exits early deadlocks the profiled one). No host-side collectives in
the launch loop, for the same reason.
"""

import argparse
import os
import sys

import torch
import torch.distributed as dist

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import deep_gemm
from deep_gemm.utils import per_token_cast_to_fp8
from deep_gemm.utils.dist import init_dist

sys.path.insert(0, os.path.join(REPO_ROOT, 'tests'))
from test_mega_moe_sm90 import _quantize_grouped_fp8_block_128_128
from test_shared_expert_sm90 import _quantize_2d_fp8_block_128_128

H, IH = 6144, 2048
E, K = 256, 8


def build_and_run(rank_idx, num_ranks, group, tokens, ns, peer_iters):
    Epr = E // num_ranks
    shared_ih = IH * ns
    torch.manual_seed(4200 + rank_idx)

    x_bf = torch.randn(tokens, H, dtype=torch.bfloat16, device='cuda')
    x_fp8, x_sf = per_token_cast_to_fp8(x_bf, use_ue8m0=False, gran_k=128,
                                        use_packed_ue8m0=False)
    l1 = _quantize_grouped_fp8_block_128_128(
        torch.randn(Epr, 2 * IH, H, dtype=torch.bfloat16, device='cuda') * 0.05)
    l2 = _quantize_grouped_fp8_block_128_128(
        torch.randn(Epr, H, IH, dtype=torch.bfloat16, device='cuda') * 0.05)

    if ns:
        s1 = _quantize_2d_fp8_block_128_128(
            torch.randn(2 * shared_ih, H, dtype=torch.bfloat16, device='cuda') * 0.05)
        s2 = _quantize_2d_fp8_block_128_128(
            torch.randn(H, shared_ih, dtype=torch.bfloat16, device='cuda') * 0.05)
        t_l1, t_l2, t_s1, t_s2 = deep_gemm.transform_weights_for_mega_moe_sm90(
            l1, l2, s1, s2)
    else:
        t_l1, t_l2 = deep_gemm.transform_weights_for_mega_moe_sm90(l1, l2)
        t_s1 = t_s2 = None

    scores = torch.randn(tokens, E, dtype=torch.float, device='cuda')
    topk_w, topk_idx = torch.topk(scores, K, dim=-1, largest=True, sorted=False)

    buf = deep_gemm.get_symm_buffer_for_mega_moe(
        group, E, tokens, K, H, IH, num_shared_experts=ns)
    cum = torch.zeros(Epr, dtype=torch.int, device='cuda')
    y = torch.empty(tokens, H, dtype=torch.bfloat16, device='cuda')

    buf.x[:tokens].copy_(x_fp8)
    buf.x_sf[:tokens].copy_(x_sf)
    buf.topk_idx[:tokens].copy_(topk_idx)
    buf.topk_weights[:tokens].copy_(topk_w)

    def launch():
        deep_gemm.fp8_mega_moe(
            y, t_l1, t_l2, buf,
            shared_l1_weights=t_s1, shared_l2_weights=t_s2,
            cumulative_local_expert_recv_stats=cum,
            recipe=(128, 128, 128), activation='swiglu',
            activation_clamp=10.0, fast_math=True)

    return launch, buf


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--num-processes', type=int, default=8)
    p.add_argument('--local-rank-idx', type=int, default=None)
    p.add_argument('--ncu-profile-only', action='store_true')
    p.add_argument('--ncu-rank', type=int, default=0)
    p.add_argument('--ncu-peer-iters', type=int, default=2000)
    p.add_argument('--tokens', type=int, default=2048)
    p.add_argument('--ns', type=int, default=1)
    args = p.parse_args()

    if args.local_rank_idx is None:
        torch.multiprocessing.spawn(_worker, args=(args.num_processes, args),
                                    nprocs=args.num_processes)
    else:
        _worker(args.local_rank_idx, args.num_processes, args)


def _worker(local_rank, num_local_ranks, args):
    rank_idx, num_ranks, group = init_dist(local_rank, num_local_ranks)
    launch, buf = build_and_run(rank_idx, num_ranks, group,
                                args.tokens, args.ns, args.ncu_peer_iters)

    if args.ncu_profile_only:
        # Profiled rank: one launch (ncu app-replay reruns it). Peers: loop so they
        # stay live across every replay. No dist.barrier() here - ncu only replays
        # the kernel, and a host collective would mismatch and deadlock.
        if rank_idx == args.ncu_rank:
            launch()
            torch.cuda.synchronize()
        else:
            for _ in range(args.ncu_peer_iters):
                launch()
            torch.cuda.synchronize()
    else:
        for _ in range(3):
            launch()
        torch.cuda.synchronize()
        dist.barrier()
        if rank_idx == 0:
            print(f'ok tokens={args.tokens} ns={args.ns}')

    buf.destroy()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
