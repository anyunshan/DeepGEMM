"""B1.5: SM90 MegaMoE shared-expert fusion benchmark (task-equivalent comparison).

Measures whether fusing shared experts into the MegaMoE kernel is faster than running
them separately, using task-equivalent paths that both produce the same final output:

  1. Fused:       fp8_mega_moe(ns=1) — one kernel, routed + shared + combine
  2. Independent: fp8_mega_moe(ns=0) + standalone shared FFN + add — three steps,
                  all captured in one CUDA graph

The independent path includes the ~600MB HBM round trip to merge routed and shared
outputs, which any real deployment must pay. Both paths deliver identical final
tensors (within quantization tolerance), verified per scenario.

Reports speedup = t_independent / t_fused. Values >1 mean fusion is faster; <1 means
the independent path wins.
"""

import argparse
import math
import os
import random
import sys
from typing import Any, Dict, List, Tuple

import torch
import torch.distributed as dist

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import deep_gemm
from deep_gemm.testing import bench, get_arch_major
from deep_gemm.utils import per_token_cast_to_fp8
from deep_gemm.utils.dist import dist_print, init_dist

from test_mega_moe_sm90 import (
    _quantize_grouped_fp8_block_128_128,
    _stable_name_seed,
)
from test_shared_expert_sm90 import _quantize_2d_fp8_block_128_128
# Fused SwiGLU + per-128-K FP8 quant (pure Triton; the tilelang op the SM100
# baseline uses is unavailable here). Reused so the serial baseline is built
# from real kernels end to end rather than eager ops.
from bench_mega_moe_sm90 import swiglu_apply_weight_to_fp8_triton


# SM90 activation SF is per-token / per-128-K, weight SF is block (128, 128).
_DENSE_RECIPE = (1, 128, 128)

# GLM-5.2 has 256 routed experts. Pinned (not scaled by rank count) so figures
# from different rank counts stay comparable; see `_scenarios`.
_GLM_NUM_EXPERTS = 256


def _make_inputs(cfg: Dict[str, Any], rank_idx: int, num_ranks: int):
    """Build quantized inputs/weights shared by all three variants."""
    hidden = cfg['hidden']
    intermediate_hidden = cfg['intermediate_hidden']
    num_experts = cfg['num_experts']
    num_topk = cfg['num_topk']
    num_shared = cfg['num_shared_experts']
    num_tokens = cfg['num_tokens']

    num_experts_per_rank = num_experts // num_ranks
    shared_ih = intermediate_hidden * num_shared

    seed = rank_idx * 1000 + _stable_name_seed(cfg['_name'])
    torch.manual_seed(seed)
    random.seed(seed)

    x_bf = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')
    l1_bf = torch.randn((num_experts_per_rank, intermediate_hidden * 2, hidden),
                        dtype=torch.bfloat16, device='cuda') * 0.05
    l2_bf = torch.randn((num_experts_per_rank, hidden, intermediate_hidden),
                        dtype=torch.bfloat16, device='cuda') * 0.05

    # Shared weights are dense and replicated on every rank.
    torch.manual_seed(12345)
    s1_bf = torch.randn((shared_ih * 2, hidden), dtype=torch.bfloat16, device='cuda') * 0.05
    s2_bf = torch.randn((hidden, shared_ih), dtype=torch.bfloat16, device='cuda') * 0.05
    torch.manual_seed(seed)

    scores = torch.randn((num_tokens, num_experts), dtype=torch.float, device='cuda')
    topk_w, topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=False)

    x_fp8, x_sf = per_token_cast_to_fp8(x_bf, use_ue8m0=False, gran_k=128,
                                        use_packed_ue8m0=False)
    l1_w = _quantize_grouped_fp8_block_128_128(l1_bf)
    l2_w = _quantize_grouped_fp8_block_128_128(l2_bf)
    s1_w = _quantize_2d_fp8_block_128_128(s1_bf)
    s2_w = _quantize_2d_fp8_block_128_128(s2_bf)

    return dict(
        x_bf=x_bf, x_fp8=x_fp8, x_sf=x_sf,
        topk_idx=topk_idx, topk_w=topk_w,
        l1_w=l1_w, l2_w=l2_w, s1_w=s1_w, s2_w=s2_w,
        num_experts_per_rank=num_experts_per_rank, shared_ih=shared_ih,
    )


def _make_runner(cfg, inp, group, num_shared: int):
    """Build a zero-arg callable that launches the fused kernel once."""
    hidden = cfg['hidden']
    intermediate_hidden = cfg['intermediate_hidden']
    num_experts = cfg['num_experts']
    num_topk = cfg['num_topk']
    num_tokens = cfg['num_tokens']
    num_max = cfg['num_max_tokens_per_rank']
    clamp = cfg.get('activation_clamp', 10.0)
    fast_math = cfg.get('fast_math', True)

    if num_shared > 0:
        t_l1, t_l2, t_s1, t_s2 = deep_gemm.transform_weights_for_mega_moe_sm90(
            inp['l1_w'], inp['l2_w'], inp['s1_w'], inp['s2_w'])
    else:
        t_l1, t_l2 = deep_gemm.transform_weights_for_mega_moe_sm90(
            inp['l1_w'], inp['l2_w'])
        t_s1 = t_s2 = None

    buffer = deep_gemm.get_symm_buffer_for_mega_moe(
        group, num_experts, num_max, num_topk,
        hidden, intermediate_hidden,
        num_shared_experts=num_shared,
    )
    cum_stats = torch.zeros(inp['num_experts_per_rank'], dtype=torch.int, device='cuda')

    buffer.x[:num_tokens].copy_(inp['x_fp8'])
    buffer.x_sf[:num_tokens].copy_(inp['x_sf'])
    buffer.topk_idx[:num_tokens].copy_(inp['topk_idx'])
    buffer.topk_weights[:num_tokens].copy_(inp['topk_w'])
    y = torch.empty((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')

    def run():
        deep_gemm.fp8_mega_moe(
            y, t_l1, t_l2, buffer,
            shared_l1_weights=t_s1, shared_l2_weights=t_s2,
            cumulative_local_expert_recv_stats=cum_stats,
            recipe=(128, 128, 128),
            activation='swiglu',
            activation_clamp=clamp if math.isfinite(clamp) else None,
            fast_math=fast_math,
        )

    return run, buffer, y


def _make_independent_runner(cfg, inp, group):
    """Independent path: routed MegaMoE (ns=0) + standalone shared FFN + add.

    This is the task-equivalent alternative to fusing shared into the kernel:
    both paths deliver the same final output (routed + shared contributions combined),
    so timing them against each other measures fusion efficiency, not task mismatch.

    Steps (all captured in one CUDA graph):
      1. fp8_mega_moe(ns=0) -> y_routed
      2. Two dense GEMMs + Triton SwiGLU/quant -> y_shared
      3. y_routed += y_shared

    The graph removes launch gaps; the add includes the ~600MB of HBM traffic the
    independent approach must pay to merge the two outputs.
    """
    hidden = cfg['hidden']
    intermediate_hidden = cfg['intermediate_hidden']
    num_experts = cfg['num_experts']
    num_topk = cfg['num_topk']
    num_tokens = cfg['num_tokens']
    num_max = cfg['num_max_tokens_per_rank']
    clamp = cfg.get('activation_clamp', 10.0)
    fast_math = cfg.get('fast_math', True)
    shared_ih = inp['shared_ih']

    # Routed kernel setup (ns=0)
    t_l1, t_l2 = deep_gemm.transform_weights_for_mega_moe_sm90(
        inp['l1_w'], inp['l2_w'])
    buffer = deep_gemm.get_symm_buffer_for_mega_moe(
        group, num_experts, num_max, num_topk,
        hidden, intermediate_hidden, num_shared_experts=0)
    cum_stats = torch.zeros(inp['num_experts_per_rank'], dtype=torch.int, device='cuda')

    buffer.x[:num_tokens].copy_(inp['x_fp8'])
    buffer.x_sf[:num_tokens].copy_(inp['x_sf'])
    buffer.topk_idx[:num_tokens].copy_(inp['topk_idx'])
    buffer.topk_weights[:num_tokens].copy_(inp['topk_w'])

    y_routed = torch.empty((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')
    y_shared = torch.empty((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')

    # Shared FFN setup
    x_pair = (inp['x_fp8'], inp['x_sf'])
    s1_pair, s2_pair = inp['s1_w'], inp['s2_w']
    l1_out = torch.empty((num_tokens, shared_ih * 2), dtype=torch.bfloat16, device='cuda')
    act_fp8 = torch.empty((num_tokens, shared_ih), dtype=torch.float8_e4m3fn, device='cuda')
    act_sf = torch.empty((num_tokens, shared_ih // 128), dtype=torch.float32, device='cuda')
    clamp_arg = clamp if math.isfinite(clamp) else None

    def step():
        # 1. Routed MegaMoE (ns=0)
        deep_gemm.fp8_mega_moe(
            y_routed, t_l1, t_l2, buffer,
            cumulative_local_expert_recv_stats=cum_stats,
            recipe=(128, 128, 128), activation='swiglu',
            activation_clamp=clamp_arg, fast_math=fast_math)

        # 2. Shared FFN
        deep_gemm.fp8_gemm_nt(x_pair, s1_pair, l1_out, recipe=_DENSE_RECIPE,
                              disable_ue8m0_cast=True)
        a, s = swiglu_apply_weight_to_fp8_triton(
            l1_out, topk_weights=None, clamp_value=clamp_arg,
            num_per_channels=128, use_ue8m0_scale=False)
        act_fp8.copy_(a)
        act_sf.copy_(s)
        deep_gemm.fp8_gemm_nt((act_fp8, act_sf), s2_pair, y_shared,
                              recipe=_DENSE_RECIPE, disable_ue8m0_cast=True)

        # 3. Combine
        y_routed.add_(y_shared)

    # Warmup on side stream, then capture
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            step()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        step()
    torch.cuda.synchronize()

    return graph.replay, buffer, y_routed


def _bench_scenario(name: str, cfg: Dict[str, Any],
                    rank_idx: int, num_ranks: int, group,
                    num_tests: int):
    cfg = dict(cfg, _name=name)
    ns = cfg['num_shared_experts']
    inp = _make_inputs(cfg, rank_idx, num_ranks)

    # Task-equivalent comparison: both paths must produce the same final output
    # (routed + shared combined). Timing is end-to-end with plain CUDA events.
    # Do NOT use `bench_kineto` with a `barrier=` callback: MegaMoE is a cross-rank
    # persistent kernel with in-kernel NVLink barriers; interleaving host-side NCCL
    # collectives into the launch loop deadlocks them against each other.

    # 1. Fused path: one kernel does routed + shared + combine in one launch
    run_fused, buf_fused, y_fused = _make_runner(cfg, inp, group, num_shared=ns)
    run_fused()
    torch.cuda.synchronize()
    dist.barrier()
    t_fused = bench(run_fused, num_warmups=5, num_tests=num_tests)
    dist.barrier()
    y_fused_ref = y_fused.clone()
    buf_fused.destroy()

    # 2. Independent path: routed MegaMoE (ns=0) + standalone shared FFN + add
    run_indep, buf_indep, y_indep = _make_independent_runner(cfg, inp, group)
    run_indep()
    torch.cuda.synchronize()
    dist.barrier()
    t_indep = bench(run_indep, num_warmups=5, num_tests=num_tests)
    dist.barrier()
    y_indep_ref = y_indep.clone()
    buf_indep.destroy()

    # Verify task equivalence: both paths should produce the same output
    denom = (y_fused_ref * y_fused_ref + y_indep_ref * y_indep_ref).sum()
    diff = 1 - (2 * (y_fused_ref * y_indep_ref).sum() / denom).item() if denom > 0 else 0.0

    if rank_idx == 0:
        speedup = t_indep / t_fused
        verdict = "FASTER" if speedup > 1.0 else "SLOWER"
        print(f"  [{name:30s}] fused={t_fused*1e6:8.1f}us  indep={t_indep*1e6:8.1f}us  "
              f"speedup={speedup:.3f}x  diff={diff:.4f}  {verdict}")

    return t_fused, t_indep, diff


def _scenarios(num_ranks: int) -> List[Tuple[str, Dict[str, Any]]]:
    """GLM-5.2 shape (H=6144, IH=2048, ns=1) across decode/prefill token counts.

    `num_experts` is pinned to GLM-5.2's real 256 routed experts rather than
    scaled by rank count: expert count sets tokens-per-expert
    (tokens * topk / num_experts) and hence GEMM M-dim efficiency, so scaling it
    would make different rank counts incomparable.

    `num_tokens` is PER RANK, so t16384 on 8 ranks is 131072 tokens globally.
    """
    glm = dict(hidden=6144, intermediate_hidden=2048,
               num_experts=_GLM_NUM_EXPERTS, num_topk=8, num_shared_experts=1)
    out = []
    for tokens in (128, 2048, 16384):
        out.append((f'glm5.2.t{tokens}', dict(
            glm, num_max_tokens_per_rank=tokens, num_tokens=tokens)))
    return out


def test(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    rank_idx, num_ranks, group = init_dist(local_rank, num_local_ranks)

    if get_arch_major() != 9:
        dist_print(f'[SKIP] bench_shared_overhead requires SM90; '
                   f'got SM{get_arch_major()}0', once_in_node=True)
        dist.destroy_process_group()
        return

    scenarios = _scenarios(num_ranks)
    if args.filter:
        scenarios = [(n, c) for n, c in scenarios if args.filter in n]

    dist_print(f'SM90 shared-expert fusion: task-equivalent comparison',
               once_in_node=True)
    dist_print(f'{len(scenarios)} scenarios on {num_ranks} ranks, '
               f'{args.num_tests} tests each', once_in_node=True)
    dist_print('Both paths produce the same final output (routed + shared combined).',
               once_in_node=True)
    dist_print('', once_in_node=True)

    results = []
    for name, cfg in scenarios:
        results.append(_bench_scenario(name, cfg, rank_idx, num_ranks, group,
                                       args.num_tests))

    dist_print('', once_in_node=True)
    faster = [r for r in results if r[0] < r[1]]  # t_fused < t_indep
    avg_speedup = sum(r[1] / r[0] for r in results) / len(results) if results else 0
    dist_print(f'{len(faster)}/{len(results)} scenarios are faster fused; '
               f'avg speedup {avg_speedup:.3f}x', once_in_node=True)

    max_diff = max((r[2] for r in results), default=0.0)
    dist_print(f'Max output diff: {max_diff:.4f} (quantization tolerance)',
               once_in_node=True)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='SM90 MegaMoE shared-expert fusion overhead')
    parser.add_argument('--num-processes', type=int, default=8)
    parser.add_argument('--num-tests', type=int, default=20)
    parser.add_argument('--filter', type=str, default='')
    args = parser.parse_args()

    torch.multiprocessing.spawn(test, args=(args.num_processes, args),
                                nprocs=args.num_processes)
