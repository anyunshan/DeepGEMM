"""Isolate why fusing shared experts costs more than the shared math itself.

probe_phases showed fusion+ (fused - routed) exceeding the standalone cost of
the shared GEMMs, and growing faster than the shared work. Two candidate causes:

  H1 fixed phase-switch cost: draining/refilling the shared smem pipeline once
     per launch. Should be roughly CONSTANT in routed size and scale only with
     shared size.

  H2 pipeline interference: shared and routed share the same kNumStages smem
     stages and full/empty barriers, so routed restarts cold and keeps paying
     per tile. Should scale with ROUTED size even at fixed shared size.

Experiments:
  E1  fix tokens, sweep ns (1,2,4)      -> shared work grows, routed fixed
  E2  fix ns=1, sweep tokens            -> routed work grows, shared grows too
  E3  fix ns=1, sweep num_experts       -> routed work grows, shared FIXED
                                           (the discriminating test)

E3 is the one that separates H1 from H2: shared cost is identical across its
rows, so any growth in fusion+ must come from routed-side interference.
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
from deep_gemm.testing import bench, get_arch_major
from deep_gemm.utils import per_token_cast_to_fp8
from deep_gemm.utils.dist import dist_print, init_dist

sys.path.insert(0, os.path.join(REPO_ROOT, 'tests'))
from test_mega_moe_sm90 import _quantize_grouped_fp8_block_128_128
from test_shared_expert_sm90 import _quantize_2d_fp8_block_128_128

_DENSE_RECIPE = (1, 128, 128)
H, IH = 6144, 2048


def _build(tokens, num_experts, num_topk, ns, rank_idx, num_ranks):
    Epr = num_experts // num_ranks
    shared_ih = IH * ns
    torch.manual_seed(7000 + rank_idx)

    x_bf = torch.randn(tokens, H, dtype=torch.bfloat16, device='cuda')
    x_fp8, x_sf = per_token_cast_to_fp8(x_bf, use_ue8m0=False, gran_k=128,
                                       use_packed_ue8m0=False)
    l1 = _quantize_grouped_fp8_block_128_128(
        torch.randn(Epr, 2 * IH, H, dtype=torch.bfloat16, device='cuda') * 0.05)
    l2 = _quantize_grouped_fp8_block_128_128(
        torch.randn(Epr, H, IH, dtype=torch.bfloat16, device='cuda') * 0.05)
    s1 = _quantize_2d_fp8_block_128_128(
        torch.randn(2 * shared_ih, H, dtype=torch.bfloat16, device='cuda') * 0.05)
    s2 = _quantize_2d_fp8_block_128_128(
        torch.randn(H, shared_ih, dtype=torch.bfloat16, device='cuda') * 0.05)

    scores = torch.randn(tokens, num_experts, dtype=torch.float, device='cuda')
    topk_w, topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=False)
    return dict(x_fp8=x_fp8, x_sf=x_sf, topk_idx=topk_idx, topk_w=topk_w,
                l1=l1, l2=l2, s1=s1, s2=s2, Epr=Epr, shared_ih=shared_ih)


def _t_mega(tokens, num_experts, num_topk, ns, d, group, n_tests):
    if ns:
        t_l1, t_l2, t_s1, t_s2 = deep_gemm.transform_weights_for_mega_moe_sm90(
            d['l1'], d['l2'], d['s1'], d['s2'])
    else:
        t_l1, t_l2 = deep_gemm.transform_weights_for_mega_moe_sm90(d['l1'], d['l2'])
        t_s1 = t_s2 = None

    buf = deep_gemm.get_symm_buffer_for_mega_moe(
        group, num_experts, tokens, num_topk, H, IH, num_shared_experts=ns)
    cum = torch.zeros(d['Epr'], dtype=torch.int, device='cuda')
    y = torch.empty(tokens, H, dtype=torch.bfloat16, device='cuda')

    def run():
        buf.x[:tokens].copy_(d['x_fp8'])
        buf.x_sf[:tokens].copy_(d['x_sf'])
        buf.topk_idx[:tokens].copy_(d['topk_idx'])
        buf.topk_weights[:tokens].copy_(d['topk_w'])
        deep_gemm.fp8_mega_moe(
            y, t_l1, t_l2, buf,
            shared_l1_weights=t_s1, shared_l2_weights=t_s2,
            cumulative_local_expert_recv_stats=cum,
            recipe=(128, 128, 128), activation='swiglu',
            activation_clamp=10.0, fast_math=True)

    run()
    torch.cuda.synchronize()
    dist.barrier()
    t = bench(run, num_warmups=5, num_tests=n_tests)
    dist.barrier()
    buf.destroy()
    return t * 1e6


def _t_shared(tokens, d, n_tests):
    sih = d['shared_ih']
    l1_out = torch.empty(tokens, sih * 2, dtype=torch.bfloat16, device='cuda')
    a = torch.zeros(tokens, sih, dtype=torch.float8_e4m3fn, device='cuda')
    a_sf = torch.ones(tokens, sih // 128, dtype=torch.float32, device='cuda')
    y = torch.empty(tokens, H, dtype=torch.bfloat16, device='cuda')

    def r1():
        deep_gemm.fp8_gemm_nt((d['x_fp8'], d['x_sf']), d['s1'], l1_out,
                              recipe=_DENSE_RECIPE, disable_ue8m0_cast=True)

    def r2():
        deep_gemm.fp8_gemm_nt((a, a_sf), d['s2'], y,
                              recipe=_DENSE_RECIPE, disable_ue8m0_cast=True)

    r1(); r2()
    torch.cuda.synchronize()
    return (bench(r1, num_warmups=5, num_tests=n_tests) +
            bench(r2, num_warmups=5, num_tests=n_tests)) * 1e6


def _row(label, tokens, E, K, ns, group, n_tests):
    d = _build(tokens, E, K, ns, dist.get_rank(), dist.get_world_size())
    t_routed = _t_mega(tokens, E, K, 0, d, group, n_tests)
    t_fused = _t_mega(tokens, E, K, ns, d, group, n_tests)
    t_sh = _t_shared(tokens, d, n_tests)
    fplus = t_fused - t_routed
    ratio = fplus / t_sh if t_sh > 0 else 0.0
    dist_print(f'{label:>22} {t_routed:>8.0f}u {t_fused:>8.0f}u {fplus:>8.0f}u '
               f'{t_sh:>7.0f}u {ratio:>6.2f}x', once_in_node=True)
    return fplus, t_sh


def test(local_rank, num_local_ranks, args):
    rank_idx, num_ranks, group = init_dist(local_rank, num_local_ranks)
    if get_arch_major() != 9:
        dist_print('[SKIP] requires SM90', once_in_node=True)
        dist.destroy_process_group()
        return

    n = args.num_tests
    hdr = (f'{"config":>22} {"routed":>9} {"fused":>9} {"fusion+":>9} '
           f'{"shared":>8} {"ratio":>7}')

    dist_print(f'Isolation probe, {num_ranks} ranks, {n} tests each', once_in_node=True)
    dist_print('ratio = fusion+ / shared-standalone (1.0 = fusion is free)',
               once_in_node=True)

    dist_print('', once_in_node=True)
    dist_print('E1: fixed tokens=2048, sweep ns (shared work grows, routed fixed)',
               once_in_node=True)
    dist_print(hdr, once_in_node=True)
    for ns in (1, 2, 4):
        _row(f't2048.ns{ns}', 2048, 256, 8, ns, group, n)

    dist_print('', once_in_node=True)
    dist_print('E2: fixed ns=1, sweep tokens (both grow)', once_in_node=True)
    dist_print(hdr, once_in_node=True)
    for tk in (512, 2048, 8192):
        _row(f't{tk}.ns1', tk, 256, 8, 1, group, n)

    dist_print('', once_in_node=True)
    dist_print('E3 (discriminating): fixed ns=1 AND fixed tokens=2048,',
               once_in_node=True)
    dist_print('    sweep num_experts -> routed GEMM work grows, shared IDENTICAL',
               once_in_node=True)
    dist_print(hdr, once_in_node=True)
    for E in (64, 128, 256):
        _row(f'E{E}.t2048.ns1', 2048, E, 8, 1, group, n)

    dist_print('', once_in_node=True)
    dist_print('Reading: if E3 ratio stays flat -> fixed phase-switch cost (H1).',
               once_in_node=True)
    dist_print('         if E3 ratio climbs    -> routed-side interference (H2).',
               once_in_node=True)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--num-processes', type=int, default=8)
    p.add_argument('--num-tests', type=int, default=15)
    args = p.parse_args()
    torch.multiprocessing.spawn(test, args=(args.num_processes, args),
                                nprocs=args.num_processes)

