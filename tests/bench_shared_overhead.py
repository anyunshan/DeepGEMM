"""B1.5: SM90 MegaMoE shared-expert fusion overhead benchmark.

Measures what the fused shared expert actually costs, against the serial
alternative you would otherwise pay:

  1. routed-only     — ns=0 kernel (the floor)
  2. fused           — ns>0 kernel, shared L1/L2 folded into the same launch
  3. serial baseline — ns=0 kernel + standalone shared FFN built from
                       DeepGEMM's own dense `fp8_gemm_nt` (NOT PyTorch matmul,
                       so the comparison is kernel-vs-kernel and does not
                       flatter the fused path)

Known bias in (3): the two GEMMs are DeepGEMM kernels, but the SwiGLU +
requantize step between them is eager PyTorch, which is slower than a fused
op would be. That inflates the serial baseline and therefore *understates*
the ratio — the real figure is worse than what this prints. Replace that
middle step (Triton, or the tilelang op the SM100 baseline uses) before
treating the ratio as a reportable number.

Design goal (doc D2): shared L1 is issued after the 320-thread rendezvous but
before the dispatch-count spin, so its WGMMA overlaps the dispatch pull. If
that overlap works, `fused - routed` should land well under the standalone
shared FFN time.
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


# SM90 activation SF is per-token / per-128-K, weight SF is block (128, 128).
_DENSE_RECIPE = (1, 128, 128)


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

    return run, buffer


def _make_shared_ffn_runner(cfg, inp):
    """Standalone shared FFN via DeepGEMM dense GEMMs (the serial alternative).

    x -> fp8_gemm_nt(s1) -> SwiGLU -> requant per-128-K -> fp8_gemm_nt(s2)

    The SwiGLU + requant step is plain PyTorch here. On SM100 the equivalent
    baseline uses a tilelang op; SM90 has no such dependency in this repo, so
    the two GEMMs (the part that dominates) are DeepGEMM kernels and only the
    elementwise middle is eager. That makes this baseline slightly pessimistic
    on the elementwise portion — noted where results are reported.
    """
    hidden = cfg['hidden']
    clamp = cfg.get('activation_clamp', 10.0)
    num_tokens = cfg['num_tokens']
    shared_ih = inp['shared_ih']

    x_pair = (inp['x_fp8'], inp['x_sf'])
    s1_pair, s2_pair = inp['s1_w'], inp['s2_w']

    l1_out = torch.empty((num_tokens, shared_ih * 2), dtype=torch.bfloat16, device='cuda')
    y = torch.empty((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')

    def run():
        deep_gemm.fp8_gemm_nt(x_pair, s1_pair, l1_out, recipe=_DENSE_RECIPE,
                              disable_ue8m0_cast=True)
        gate, up = l1_out.chunk(2, dim=-1)
        if math.isfinite(clamp):
            gate = gate.clamp(max=clamp)
            up = up.clamp(min=-clamp, max=clamp)
        act = torch.nn.functional.silu(gate.float()) * up.float()
        act_fp8, act_sf = per_token_cast_to_fp8(
            act.to(torch.bfloat16), use_ue8m0=False, gran_k=128,
            use_packed_ue8m0=False)
        deep_gemm.fp8_gemm_nt((act_fp8, act_sf), s2_pair, y, recipe=_DENSE_RECIPE,
                              disable_ue8m0_cast=True)

    return run


def _bench_scenario(name: str, cfg: Dict[str, Any],
                    rank_idx: int, num_ranks: int, group,
                    num_tests: int):
    cfg = dict(cfg, _name=name)
    ns = cfg['num_shared_experts']
    inp = _make_inputs(cfg, rank_idx, num_ranks)

    # All three variants are timed with plain CUDA events (`bench`), end-to-end,
    # exactly like `bench_mega_moe_sm90.py`. Do NOT use `bench_kineto` with a
    # `barrier=` callback here: MegaMoE is a cross-rank persistent kernel with its
    # own in-kernel NVLink barrier, and interleaving a host-side NCCL collective
    # into the launch loop deadlocks the two synchronisation schemes against each
    # other (observed: both ranks parked on dist.barrier, GPU spinning at 100%).
    # `dist.barrier()` is only ever called OUTSIDE a timed region, to line the
    # ranks up before and after each measurement.

    # 1. routed-only floor
    run_routed, buf_routed = _make_runner(cfg, inp, group, num_shared=0)
    run_routed()
    torch.cuda.synchronize()
    dist.barrier()
    t_routed = bench(run_routed, num_warmups=5, num_tests=num_tests)
    dist.barrier()
    buf_routed.destroy()

    # 2. fused
    run_fused, buf_fused = _make_runner(cfg, inp, group, num_shared=ns)
    run_fused()
    torch.cuda.synchronize()
    dist.barrier()
    t_fused = bench(run_fused, num_warmups=5, num_tests=num_tests)
    dist.barrier()
    buf_fused.destroy()

    # 3. standalone shared FFN (DeepGEMM dense GEMMs + eager SwiGLU/requant).
    # End-to-end time for the whole sequence, which is the right yardstick: it is
    # what you would actually pay to run the shared expert separately.
    run_shared = _make_shared_ffn_runner(cfg, inp)
    run_shared()
    torch.cuda.synchronize()
    dist.barrier()
    t_shared_total = bench(run_shared, num_warmups=5, num_tests=num_tests)
    dist.barrier()

    fused_delta = t_fused - t_routed
    ratio = fused_delta / t_shared_total if t_shared_total > 0 else float('nan')

    dist_print(
        f'  [{name:<22}] routed={t_routed * 1e6:8.1f}us  fused={t_fused * 1e6:8.1f}us  '
        f'delta={fused_delta * 1e6:7.1f}us  serial_shared={t_shared_total * 1e6:7.1f}us  '
        f'ratio={ratio * 100:5.1f}%  {"OK" if ratio < 0.5 else "--"}',
        once_in_node=True)

    return dict(name=name, routed=t_routed, fused=t_fused,
                delta=fused_delta, shared=t_shared_total, ratio=ratio)


def _scenarios(num_ranks: int) -> List[Tuple[str, Dict[str, Any]]]:
    """GLM-5.2 shape (H=6144, IH=2048, ns=1) across decode/prefill token counts.

    CAVEAT: `num_experts` scales with rank count here (inherited from the
    correctness-test scenario table, where it lets small configs run at any rank
    count). For perf that is wrong — expert count sets tokens-per-expert
    (tokens * topk / num_experts) and hence the GEMM M-dim efficiency, so
    numbers from different rank counts are NOT comparable. GLM-5.2 really has
    256 routed experts; pin it before quoting figures.

    `num_tokens` is PER RANK, so t16384 on 8 ranks is 131072 tokens globally.
    """
    glm = dict(hidden=6144, intermediate_hidden=2048,
               num_experts=8 * num_ranks, num_topk=8, num_shared_experts=1)
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

    dist_print(f'SM90 shared-expert fusion overhead: {len(scenarios)} scenarios '
               f'on {num_ranks} ranks, {args.num_tests} tests each',
               once_in_node=True)
    dist_print('  ratio = (fused - routed) / standalone_shared_ffn; '
               'design goal < 50%', once_in_node=True)

    results = []
    for name, cfg in scenarios:
        results.append(_bench_scenario(name, cfg, rank_idx, num_ranks, group,
                                       args.num_tests))

    dist_print('', once_in_node=True)
    good = [r for r in results if r['ratio'] < 0.5]
    dist_print(f'{len(good)}/{len(results)} scenarios meet the <50% overlap goal',
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
