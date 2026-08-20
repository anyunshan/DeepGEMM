"""SM90 MegaMoE shared-expert correctness test.

`test_mega_moe_sm90.py` never sets `num_shared_experts`, so the fused shared
L1/L2 phases are compile-time dead in every one of its 32 scenarios. This test
exercises them: shared L1 reads `x` directly (zero-copy alias, no dispatch),
SwiGLU feeds the shared L2, and the shared result enters combine with weight
1.0 alongside the routed topk contributions.

Reference = routed reference (reused verbatim) + dense shared path.
"""

import argparse
import math
import random
import sys
from typing import Any, Dict, List, Tuple

import torch
import torch.distributed as dist

import deep_gemm
from deep_gemm.testing import calc_diff, get_arch_major
from deep_gemm.utils import per_token_cast_to_fp8
from deep_gemm.utils.dist import dist_print, init_dist

from test_mega_moe_sm90 import (
    _dequant_block_128_128,
    _dequant_per_token_per_128_k,
    _quantize_grouped_fp8_block_128_128,
    _reference_fused,
    _stable_name_seed,
    _swiglu_fp32,
)


def _quantize_2d_fp8_block_128_128(w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Block-(128, 128) FP8 quantization for a 2D weight, via the grouped helper."""
    fp8, sf = _quantize_grouped_fp8_block_128_128(w.unsqueeze(0))
    return fp8.squeeze(0), sf.squeeze(0)


def _shared_reference(
    x_fp8: torch.Tensor, x_sf: torch.Tensor,
    s1_fp8: torch.Tensor, s1_sf: torch.Tensor,
    s2_fp8: torch.Tensor, s2_sf: torch.Tensor,
    intermediate_hidden: int, activation_clamp: float,
) -> torch.Tensor:
    """Dense shared-expert path in fp32: x -> L1 -> SwiGLU -> requant -> L2.

    Mirrors the kernel: the SwiGLU output is requantized to FP8 at per-128 K
    before the L2 GEMM, so the reference must round-trip through FP8 too.
    """
    if x_fp8.shape[0] == 0:
        return torch.zeros((0, s2_fp8.shape[0]), dtype=torch.float32, device='cuda')
    
    x_f32 = _dequant_per_token_per_128_k(x_fp8, x_sf)             # (M, H)
    s1_f32 = _dequant_block_128_128(s1_fp8, s1_sf)               # (2*SIH, H)
    s2_f32 = _dequant_block_128_128(s2_fp8, s2_sf)               # (H, SIH)

    gate_up = x_f32 @ s1_f32.t()                                  # (M, 2*SIH)
    act = _swiglu_fp32(gate_up, activation_clamp)                 # (M, SIH)

    act_fp8, act_sf = per_token_cast_to_fp8(
        act.to(torch.bfloat16), use_ue8m0=False, gran_k=128, use_packed_ue8m0=False)
    act_f32 = _dequant_per_token_per_128_k(act_fp8, act_sf)

    return act_f32 @ s2_f32.t()                                   # (M, H)


def _run_shared_scenario(
    name: str, cfg: Dict[str, Any],
    rank_idx: int, num_ranks: int, group: dist.ProcessGroup,
    diff_tol: float,
):
    num_max = cfg['num_max_tokens_per_rank']
    num_tokens = cfg.get('num_tokens', num_max)
    hidden = cfg['hidden']
    intermediate_hidden = cfg['intermediate_hidden']
    num_experts = cfg['num_experts']
    num_topk = cfg['num_topk']
    num_shared = cfg['num_shared_experts']
    activation_clamp = cfg.get('activation_clamp', 10.0)
    fast_math = cfg.get('fast_math', True)

    num_experts_per_rank = num_experts // num_ranks
    shared_ih = intermediate_hidden * num_shared
    assert num_tokens <= num_max
    assert hidden % 128 == 0 and intermediate_hidden % 128 == 0
    assert shared_ih % 128 == 0

    seed = rank_idx * 1000 + _stable_name_seed(name)
    torch.manual_seed(seed)
    random.seed(seed)

    # ---- Inputs -------------------------------------------------------------
    x_bf = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')
    l1_bf = torch.randn((num_experts_per_rank, intermediate_hidden * 2, hidden),
                        dtype=torch.bfloat16, device='cuda') * 0.05
    l2_bf = torch.randn((num_experts_per_rank, hidden, intermediate_hidden),
                        dtype=torch.bfloat16, device='cuda') * 0.05
    # Shared weights are 2D and replicated on every rank (dense, not sharded).
    torch.manual_seed(12345)  # same shared weights across ranks
    s1_bf = torch.randn((shared_ih * 2, hidden), dtype=torch.bfloat16, device='cuda') * 0.05
    s2_bf = torch.randn((hidden, shared_ih), dtype=torch.bfloat16, device='cuda') * 0.05
    torch.manual_seed(seed)

    scores = torch.randn((num_tokens, num_experts), dtype=torch.float, device='cuda')
    topk_w, topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=False)

    x_fp8, x_sf = per_token_cast_to_fp8(x_bf, use_ue8m0=False, gran_k=128,
                                        use_packed_ue8m0=False)
    l1_w_fp8, l1_w_sf = _quantize_grouped_fp8_block_128_128(l1_bf)
    l2_w_fp8, l2_w_sf = _quantize_grouped_fp8_block_128_128(l2_bf)
    s1_w_fp8, s1_w_sf = _quantize_2d_fp8_block_128_128(s1_bf)
    s2_w_fp8, s2_w_sf = _quantize_2d_fp8_block_128_128(s2_bf)

    t_l1, t_l2, t_s1, t_s2 = deep_gemm.transform_weights_for_mega_moe_sm90(
        (l1_w_fp8, l1_w_sf), (l2_w_fp8, l2_w_sf),
        (s1_w_fp8, s1_w_sf), (s2_w_fp8, s2_w_sf),
    )

    buffer = deep_gemm.get_symm_buffer_for_mega_moe(
        group, num_experts, num_max, num_topk,
        hidden, intermediate_hidden,
        num_shared_experts=num_shared,
    )
    cum_stats = torch.zeros(num_experts_per_rank, dtype=torch.int, device='cuda')

    buffer.x[:num_tokens].copy_(x_fp8)
    buffer.x_sf[:num_tokens].copy_(x_sf)
    buffer.topk_idx[:num_tokens].copy_(topk_idx)
    buffer.topk_weights[:num_tokens].copy_(topk_w)
    y = torch.empty((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')
    deep_gemm.fp8_mega_moe(
        y, t_l1, t_l2, buffer,
        shared_l1_weights=t_s1, shared_l2_weights=t_s2,
        cumulative_local_expert_recv_stats=cum_stats,
        recipe=(128, 128, 128),
        activation='swiglu',
        activation_clamp=activation_clamp if math.isfinite(activation_clamp) else None,
        fast_math=fast_math,
    )
    torch.cuda.synchronize()

    # ---- Reference: routed + shared -----------------------------------------
    y_routed = _reference_fused(
        x_fp8, x_sf, topk_idx, topk_w,
        l1_w_fp8, l1_w_sf, l2_w_fp8, l2_w_sf,
        rank_idx, num_ranks, group,
        num_experts, num_topk, hidden, intermediate_hidden,
        activation_clamp, l2_act_gran_k=128,
    )
    y_shared = _shared_reference(
        x_fp8, x_sf, s1_w_fp8, s1_w_sf, s2_w_fp8, s2_w_sf,
        intermediate_hidden, activation_clamp,
    )
    y_ref = (y_routed.float() + y_shared).to(torch.bfloat16)

    diff = calc_diff(y, y_ref)
    ok = diff < diff_tol
    if rank_idx == 0:
        print(f'  [{name:<34}] diff={diff:.4f} (tol={diff_tol:.2f}) '
              f'{"OK" if ok else "FAIL"}', flush=True)

    # Guard against a silently-dead shared phase: with shared experts enabled the
    # result must differ from the routed-only reference, unless tokens=0.
    if num_tokens > 0:
        shared_mag = y_shared.abs().mean().item()
        assert shared_mag > 1e-4, f'{name}: shared reference is ~0, test is vacuous'
        routed_only_diff = calc_diff(y, y_routed)
        assert routed_only_diff > 1e-3, (
            f'{name}: output matches routed-only reference (diff={routed_only_diff:.6f}) '
            f'— shared expert phase appears to be a no-op')

    assert ok, f'{name}: diff={diff} >= tol={diff_tol}'
    buffer.destroy()
    dist.barrier()


def _scenarios(num_ranks: int) -> List[Tuple[str, Dict[str, Any]]]:
    out: List[Tuple[str, Dict[str, Any]]] = []
    # Smoke: smallest shape that satisfies the 128 alignment on shared_ih.
    out.append(('S1.smoke.ns1', dict(
        num_max_tokens_per_rank=64, num_tokens=64,
        hidden=512, intermediate_hidden=512,
        num_experts=8 * num_ranks, num_topk=2, num_shared_experts=1)))
    # Multiple shared experts (shared_ih = IH * ns).
    for ns in (1, 2, 4, 8):
        out.append((f'S2.ns{ns}.t256', dict(
            num_max_tokens_per_rank=256, num_tokens=256,
            hidden=1024, intermediate_hidden=512,
            num_experts=8 * num_ranks, num_topk=2, num_shared_experts=ns)))
    # Token-count edges: partial tile, larger batch, and zero tokens.
    for tokens in (0, 16, 260, 1024):
        out.append((f'S3.tokens{tokens}', dict(
            num_max_tokens_per_rank=max(tokens, 64), num_tokens=tokens,
            hidden=512, intermediate_hidden=512,
            num_experts=8 * num_ranks, num_topk=2, num_shared_experts=1)))
    # Activation clamp + fast_math variations.
    out.append(('S4.clamp1.0', dict(
        num_max_tokens_per_rank=128, num_tokens=128,
        hidden=512, intermediate_hidden=512,
        num_experts=8 * num_ranks, num_topk=2, num_shared_experts=1,
        activation_clamp=1.0)))
    out.append(('S4.fm0', dict(
        num_max_tokens_per_rank=128, num_tokens=128,
        hidden=512, intermediate_hidden=512,
        num_experts=8 * num_ranks, num_topk=2, num_shared_experts=1,
        fast_math=False)))
    return out


def test(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    rank_idx, num_ranks, group = init_dist(local_rank, num_local_ranks)

    if get_arch_major() != 9:
        dist_print(f'[SKIP] test_shared_expert_sm90 requires SM90; '
                   f'got SM{get_arch_major()}0', once_in_node=True)
        dist.destroy_process_group()
        return

    scenarios = _scenarios(num_ranks)
    if args.filter:
        scenarios = [(n, c) for n, c in scenarios if args.filter in n]

    dist_print(f'SM90 MegaMoE shared-expert test: {len(scenarios)} scenarios '
               f'on {num_ranks} ranks', once_in_node=True)

    failures: List[str] = []
    for name, cfg in scenarios:
        try:
            _run_shared_scenario(name, cfg, rank_idx, num_ranks, group, args.diff_tol)
        except AssertionError as ex:
            dist_print(f'  [{name}] FAIL: {ex}', once_in_node=True)
            failures.append(name)
            if args.fail_fast:
                break

    dist_print('', once_in_node=True)
    if failures:
        dist_print(f'FAILED {len(failures)}/{len(scenarios)} scenarios: {failures}',
                   once_in_node=True)
    else:
        dist_print(f'PASSED all {len(scenarios)} shared-expert scenarios',
                   once_in_node=True)

    dist.barrier()
    dist.destroy_process_group()
    if failures:
        sys.exit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SM90 MegaMoE shared-expert tests')
    parser.add_argument('--num-processes', type=int, default=2)
    parser.add_argument('--filter', type=str, default='')
    parser.add_argument('--diff-tol', type=float, default=0.01)
    parser.add_argument('--fail-fast', action='store_true')
    args = parser.parse_args()

    torch.multiprocessing.spawn(test, args=(args.num_processes, args),
                                nprocs=args.num_processes)
