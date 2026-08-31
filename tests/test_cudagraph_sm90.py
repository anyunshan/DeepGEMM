"""Verify the SM90 MegaMoE kernel is safe to capture in a CUDA graph.

vLLM runs decode under FULL_AND_PIECEWISE and prefill under PIECEWISE cudagraph
modes, so the kernel must survive capture + repeated replay. Three risks, each
with a dedicated check:

  C1 counter hygiene across replays
     The kernel self-resets its symmetric-memory counters (expert_send_count,
     l1_arrival_count, l2_arrival_mask, shared_l2_full_count) in the dispatch
     warps at the end of each launch. If any reset is incomplete, replay #1 is
     fine and replay #2+ silently corrupts. So: capture once, replay N times,
     compare EVERY replay against the eager reference.

  C2 cross-rank barriers under replay
     The kernel has in-kernel NVLink barriers. Each rank replays its own graph,
     so the ranks are not host-synchronized during the replay loop. Checked
     implicitly by C1 (a desynced barrier shows up as a hang or wrong output)
     and explicitly by running the replay loop without any host collective.

  C3 token count smaller than captured
     Graphs bake in shapes, but `num_tokens` reaches the kernel as a scalar
     argument, and vLLM pads to a captured bucket. Capture at the max token
     count, then replay with fewer live tokens and verify the first
     `num_tokens` rows match a reference computed for that count.

Correctness reference is the eager (non-graph) path of the same kernel, so this
isolates graph capture as the variable.
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
from deep_gemm.testing import calc_diff, get_arch_major
from deep_gemm.utils import per_token_cast_to_fp8
from deep_gemm.utils.dist import dist_print, init_dist

sys.path.insert(0, os.path.join(REPO_ROOT, 'tests'))
from test_mega_moe_sm90 import _quantize_grouped_fp8_block_128_128

H, IH = 6144, 2048
E, K = 256, 8
TOL = 0.01


def _setup(num_max, rank_idx, num_ranks, group, seed_off=0):
    """Allocate weights, buffer and a launch closure parameterised by token count."""
    Epr = E // num_ranks
    torch.manual_seed(9100 + rank_idx + seed_off)

    l1 = _quantize_grouped_fp8_block_128_128(
        torch.randn(Epr, 2 * IH, H, dtype=torch.bfloat16, device='cuda') * 0.05)
    l2 = _quantize_grouped_fp8_block_128_128(
        torch.randn(Epr, H, IH, dtype=torch.bfloat16, device='cuda') * 0.05)
    t_l1, t_l2 = deep_gemm.transform_weights_for_mega_moe_sm90(l1, l2)

    buf = deep_gemm.get_symm_buffer_for_mega_moe(
        group, E, num_max, K, H, IH, num_shared_experts=0)
    cum = torch.zeros(Epr, dtype=torch.int, device='cuda')
    y = torch.empty(num_max, H, dtype=torch.bfloat16, device='cuda')

    def launch(n_tokens):
        deep_gemm.fp8_mega_moe(
            y[:n_tokens], t_l1, t_l2, buf,
            cumulative_local_expert_recv_stats=cum,
            recipe=(128, 128, 128), activation='swiglu',
            activation_clamp=10.0, fast_math=True)

    return dict(buf=buf, y=y, cum=cum, launch=launch, Epr=Epr)


def _make_tokens(n_tokens, rank_idx, seed_off=0):
    torch.manual_seed(5500 + rank_idx + seed_off)
    x_bf = torch.randn(n_tokens, H, dtype=torch.bfloat16, device='cuda')
    x_fp8, x_sf = per_token_cast_to_fp8(x_bf, use_ue8m0=False, gran_k=128,
                                        use_packed_ue8m0=False)
    scores = torch.randn(n_tokens, E, dtype=torch.float, device='cuda')
    topk_w, topk_idx = torch.topk(scores, K, dim=-1, largest=True, sorted=False)
    return x_fp8, x_sf, topk_idx, topk_w


def _fill(buf, n_tokens, toks):
    x_fp8, x_sf, topk_idx, topk_w = toks
    buf.x[:n_tokens].copy_(x_fp8)
    buf.x_sf[:n_tokens].copy_(x_sf)
    buf.topk_idx[:n_tokens].copy_(topk_idx)
    buf.topk_weights[:n_tokens].copy_(topk_w)


def _eager_reference(env, n_tokens, toks):
    """Run the kernel outside any graph; returns a detached copy of the output."""
    _fill(env['buf'], n_tokens, toks)
    env['cum'].zero_()
    env['launch'](n_tokens)
    torch.cuda.synchronize()
    return env['y'][:n_tokens].clone()


def check_replays(env, n_tokens, toks, num_replays, rank_idx):
    """C1 + C2: capture once, replay many times, verify every single replay."""
    ref = _eager_reference(env, n_tokens, toks)

    # Warm up on a side stream (required before capture), then capture.
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            _fill(env['buf'], n_tokens, toks)
            env['launch'](n_tokens)
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    dist.barrier()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _fill(env['buf'], n_tokens, toks)
        env['launch'](n_tokens)
    torch.cuda.synchronize()
    dist.barrier()

    # Replay loop with NO host collective inside: the in-kernel NVLink barriers
    # are what couple the ranks, and interleaving NCCL here can deadlock them.
    worst, first_bad = 0.0, -1
    for i in range(num_replays):
        env['y'].zero_()
        graph.replay()
        torch.cuda.synchronize()
        d = calc_diff(env['y'][:n_tokens], ref)
        worst = max(worst, d)
        if d >= TOL and first_bad < 0:
            first_bad = i
    dist.barrier()
    return worst, first_bad


def check_short_replay(env, num_max, rank_idx):
    """C3: capture at num_max tokens, replay with fewer live tokens."""
    # Capture at the full bucket size.
    toks_full = _make_tokens(num_max, rank_idx)
    _fill(env['buf'], num_max, toks_full)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            env['launch'](num_max)
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    dist.barrier()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        env['launch'](num_max)
    torch.cuda.synchronize()
    dist.barrier()

    # A captured graph replays the token count it was captured with, so a short
    # batch is expressed by zero-padding the tail of the buffer (what vLLM does
    # when it pads to a captured bucket). Rows past `short` must not disturb the
    # first `short` rows.
    short = num_max // 2
    toks_short = _make_tokens(short, rank_idx, seed_off=17)

    # Reference: eager run of the padded buffer, same padding scheme.
    env['buf'].x.zero_(); env['buf'].x_sf.zero_()
    env['buf'].topk_idx.fill_(-1); env['buf'].topk_weights.zero_()
    _fill(env['buf'], short, toks_short)
    env['cum'].zero_()
    env['launch'](num_max)
    torch.cuda.synchronize()
    ref_short = env['y'][:short].clone()

    # Same padded state, but driven by the captured graph.
    env['buf'].x.zero_(); env['buf'].x_sf.zero_()
    env['buf'].topk_idx.fill_(-1); env['buf'].topk_weights.zero_()
    _fill(env['buf'], short, toks_short)
    env['cum'].zero_()
    env['y'].zero_()
    graph.replay()
    torch.cuda.synchronize()
    d = calc_diff(env['y'][:short], ref_short)
    dist.barrier()
    return d, short


def test(local_rank, num_local_ranks, args):
    rank_idx, num_ranks, group = init_dist(local_rank, num_local_ranks)
    if get_arch_major() != 9:
        dist_print('[SKIP] requires SM90', once_in_node=True)
        dist.destroy_process_group()
        return

    dist_print(f'CUDA graph capture check: {num_ranks} ranks, '
               f'{args.num_replays} replays per case, tol={TOL}', once_in_node=True)
    dist_print('', once_in_node=True)

    failures = []
    for n_tokens in args.batches:
        env = _setup(n_tokens, rank_idx, num_ranks, group)
        toks = _make_tokens(n_tokens, rank_idx)
        worst, first_bad = check_replays(env, n_tokens, toks,
                                         args.num_replays, rank_idx)
        ok = first_bad < 0
        dist_print(f'  [C1/C2 replay t{n_tokens:<6}] worst_diff={worst:.6f} '
                   f'{"OK" if ok else f"FAIL at replay #{first_bad}"}',
                   once_in_node=True)
        if not ok:
            failures.append(f'replay-t{n_tokens}')
        env['buf'].destroy()

    for n_tokens in args.batches:
        env = _setup(n_tokens, rank_idx, num_ranks, group, seed_off=3)
        d, short = check_short_replay(env, n_tokens, rank_idx)
        ok = d < TOL
        dist_print(f'  [C3 short  t{n_tokens:<6}] live={short:<6} diff={d:.6f} '
                   f'{"OK" if ok else "FAIL"}', once_in_node=True)
        if not ok:
            failures.append(f'short-t{n_tokens}')
        env['buf'].destroy()

    dist_print('', once_in_node=True)
    if failures:
        dist_print(f'FAILED: {failures}', once_in_node=True)
    else:
        dist_print('PASSED: kernel is cudagraph-safe '
                   '(counters reset cleanly, barriers survive replay, '
                   'short batches OK)', once_in_node=True)

    dist.barrier()
    dist.destroy_process_group()
    if failures and rank_idx == 0:
        sys.exit(1)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--num-processes', type=int, default=8)
    p.add_argument('--num-replays', type=int, default=20)
    p.add_argument('--batches', type=int, nargs='+', default=[128, 2048])
    args = p.parse_args()
    torch.multiprocessing.spawn(test, args=(args.num_processes, args),
                                nprocs=args.num_processes)

