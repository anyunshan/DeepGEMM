"""Verify the SM90 pre-dispatch kernel against the PyTorch staging path.

The fused kernel's inputs are produced two ways in this project:

  reference: per_token_cast_to_fp8 + four copies into the buffer views
  fused:     mega_moe_pre_dispatch_sm90, one kernel

These agree numerically but NOT bit-for-bit, and that is expected. Both compute
sf = amax / 448, but the reference reaches amax via `.abs().float().amax()`
(bf16 -> fp32 before the reduction, PyTorch's ordering) while the kernel converts
per element and reduces across lanes. The results differ in the last fp32 ulp
(measured: sf 8.1612728536e-03 vs 8.1612719223e-03), and 1 ulp on sf is enough to
push elements sitting near an FP8 rounding boundary into the neighbouring bucket.
Since e4m3 steps are coarse at large magnitudes, such an element moves by a whole
bucket (measured max |dx| = 32.0 on ~0.16% of elements), even though the relative
error stays ~1e-7. End to end that lands at ~1.8e-4, far inside the 0.01 tolerance
the rest of the suite uses, so this is a precision-path difference rather than a
defect. The kernel keeps its algorithm: it matches the SGLang production version.

The reference is therefore kept as an INDEPENDENT oracle (not rewritten to mimic
the kernel), and the checks use tolerances instead of exact equality:

  P1 staging equivalence        x / x_sf within quantization tolerance
  P2 padding tail               slots past num_tokens read inactive (idx -1, w 0)
  P3 scaling factor folding     routed_scaling_factor lands in the weights (exact)
  P4 end-to-end equivalence     fp8_mega_moe output within calc_diff < 0.01
  P5 short batch                num_tokens < capacity leaves the tail inactive

topk_idx / topk_weights ARE checked exactly: they involve no quantization.
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

# Quantization-path tolerances (see module docstring). Measured values are
# ~1e-6 for staging and ~1.8e-4 end to end, so these leave headroom while
# still catching a real regression.
QUANT_TOL = 1e-3
E2E_TOL = 0.01


def _stage_reference(buf, x_bf, topk_idx, topk_w, num_tokens, rsf):
    """The PyTorch staging path the tests have been using."""
    buf.x.zero_(); buf.x_sf.zero_()
    buf.topk_idx.fill_(-1); buf.topk_weights.zero_()
    if num_tokens:
        x_fp8, x_sf = per_token_cast_to_fp8(x_bf[:num_tokens], use_ue8m0=False,
                                            gran_k=128, use_packed_ue8m0=False)
        buf.x[:num_tokens].copy_(x_fp8)
        buf.x_sf[:num_tokens].copy_(x_sf)
        buf.topk_idx[:num_tokens].copy_(topk_idx[:num_tokens].to(torch.int64))
        buf.topk_weights[:num_tokens].copy_(topk_w[:num_tokens] * rsf)


def _snapshot(buf):
    return (buf.x.clone(), buf.x_sf.clone(),
            buf.topk_idx.clone(), buf.topk_weights.clone())


def _run(local_rank, num_local_ranks, args):
    rank_idx, num_ranks, group = init_dist(local_rank, num_local_ranks)
    if get_arch_major() != 9:
        dist_print('[SKIP] requires SM90', once_in_node=True)
        dist.destroy_process_group()
        return

    Epr = E // num_ranks
    rsf = 2.5  # GLM-5.2 routed_scaling_factor
    failures = []

    dist_print(f'SM90 pre-dispatch check: {num_ranks} ranks, '
               f'routed_scaling_factor={rsf}', once_in_node=True)
    dist_print('', once_in_node=True)

    for num_max in args.batches:
        for num_tokens in (num_max, num_max // 2, 0):
            torch.manual_seed(3300 + rank_idx)
            x_bf = torch.randn(num_max, H, dtype=torch.bfloat16, device='cuda')
            scores = torch.randn(num_max, E, dtype=torch.float, device='cuda')
            topk_w, topk_idx64 = torch.topk(scores, K, dim=-1, largest=True, sorted=False)
            topk_idx = topk_idx64.to(torch.int32).contiguous()
            topk_w = topk_w.contiguous()

            buf = deep_gemm.get_symm_buffer_for_mega_moe(
                group, E, num_max, K, H, IH, num_shared_experts=0)

            # Reference staging
            _stage_reference(buf, x_bf, topk_idx64, topk_w, num_tokens, rsf)
            torch.cuda.synchronize()
            ref = _snapshot(buf)

            # Fused staging (start from a dirty buffer so the kernel must write
            # every slot it owns, including the inactive tail)
            buf.x.fill_(torch.finfo(torch.float8_e4m3fn).max)
            buf.x_sf.fill_(7.0)
            buf.topk_idx.fill_(12345)
            buf.topk_weights.fill_(9.0)
            deep_gemm.mega_moe_pre_dispatch_sm90(
                x_bf, topk_idx, topk_w,
                buf.x, buf.x_sf, buf.topk_idx, buf.topk_weights,
                num_tokens=num_tokens, group_size=128,
                routed_scaling_factor=rsf)
            torch.cuda.synchronize()
            got = _snapshot(buf)

            # P1: staging equivalence on the live rows.
            # x / x_sf go through quantization, so they are compared with a
            # tolerance (see module docstring for why bit-equality is not
            # expected). Routing data carries no quantization and is exact.
            worst = {}
            for nm, r, g in zip(('x', 'x_sf', 'topk_idx', 'topk_weights'), ref, got):
                if num_tokens == 0:
                    continue
                r_live = r[:num_tokens].float()
                g_live = g[:num_tokens].float()
                tag = f'{nm}@t{num_tokens}/{num_max}'
                if nm in ('topk_idx', 'topk_weights'):
                    if not torch.equal(r_live, g_live):
                        dist_print(f'  [P1 {tag:<22}] MISMATCH (must be exact)',
                                   once_in_node=True)
                        failures.append(f'P1-{tag}')
                else:
                    d = calc_diff(g_live, r_live)
                    worst[nm] = d
                    if d >= QUANT_TOL:
                        dist_print(f'  [P1 {tag:<22}] diff={d:.6f} '
                                   f'exceeds {QUANT_TOL}', once_in_node=True)
                        failures.append(f'P1-{tag}')

            # P2/P5: the tail must read inactive
            if num_tokens < num_max:
                tail_idx = got[2][num_tokens:]
                tail_w = got[3][num_tokens:]
                ok_tail = bool((tail_idx == -1).all()) and bool((tail_w == 0).all())
                tag = f't{num_tokens}/{num_max}'
                if not ok_tail:
                    dist_print(f'  [P2 tail {tag:<18}] MISMATCH: idx/-1 or weight/0 '
                               f'not honoured', once_in_node=True)
                    failures.append(f'P2-{tag}')

            # P3: scaling factor really folded in
            if num_tokens:
                expect = (topk_w[:num_tokens] * rsf).float()
                if not torch.allclose(got[3][:num_tokens].float(), expect,
                                      rtol=0, atol=0):
                    dist_print(f'  [P3 rsf   t{num_tokens}/{num_max}] MISMATCH',
                               once_in_node=True)
                    failures.append(f'P3-t{num_tokens}')

            qdiff = (f'x={worst.get("x", 0.0):.2e} sf={worst.get("x_sf", 0.0):.2e}'
                     if worst else 'n/a')
            dist_print(f'  [staged t{num_tokens:<6}/{num_max:<6}] quant diff {qdiff}'
                       f' | routing exact, tail inactive, rsf folded',
                       once_in_node=True)
            buf.destroy()
            dist.barrier()

    # P4: end-to-end - does the kernel produce the same y either way?
    dist_print('', once_in_node=True)
    for num_max in args.batches:
        num_tokens = num_max
        torch.manual_seed(7700 + rank_idx)
        x_bf = torch.randn(num_max, H, dtype=torch.bfloat16, device='cuda')
        scores = torch.randn(num_max, E, dtype=torch.float, device='cuda')
        topk_w, topk_idx64 = torch.topk(scores, K, dim=-1, largest=True, sorted=False)
        topk_idx = topk_idx64.to(torch.int32).contiguous()
        topk_w = topk_w.contiguous()

        l1 = _quantize_grouped_fp8_block_128_128(
            torch.randn(Epr, 2 * IH, H, dtype=torch.bfloat16, device='cuda') * 0.05)
        l2 = _quantize_grouped_fp8_block_128_128(
            torch.randn(Epr, H, IH, dtype=torch.bfloat16, device='cuda') * 0.05)
        t_l1, t_l2 = deep_gemm.transform_weights_for_mega_moe_sm90(l1, l2)

        buf = deep_gemm.get_symm_buffer_for_mega_moe(
            group, E, num_max, K, H, IH, num_shared_experts=0)
        cum = torch.zeros(Epr, dtype=torch.int, device='cuda')

        def launch():
            y = torch.empty(num_tokens, H, dtype=torch.bfloat16, device='cuda')
            deep_gemm.fp8_mega_moe(
                y, t_l1, t_l2, buf,
                cumulative_local_expert_recv_stats=cum,
                recipe=(128, 128, 128), activation='swiglu',
                activation_clamp=10.0, fast_math=True)
            return y

        _stage_reference(buf, x_bf, topk_idx64, topk_w, num_tokens, rsf)
        cum.zero_()
        y_ref = launch()
        torch.cuda.synchronize()
        dist.barrier()

        deep_gemm.mega_moe_pre_dispatch_sm90(
            x_bf, topk_idx, topk_w,
            buf.x, buf.x_sf, buf.topk_idx, buf.topk_weights,
            num_tokens=num_tokens, group_size=128, routed_scaling_factor=rsf)
        cum.zero_()
        y_got = launch()
        torch.cuda.synchronize()

        d = calc_diff(y_got, y_ref)
        ok = d < E2E_TOL
        dist_print(f'  [P4 e2e t{num_max:<6}] diff={d:.6f} (tol={E2E_TOL}) '
                   f'{"OK" if ok else "FAIL"}', once_in_node=True)
        if not ok:
            failures.append(f'P4-t{num_max}')
        buf.destroy()
        dist.barrier()

    dist_print('', once_in_node=True)
    if failures:
        dist_print(f'FAILED: {failures}', once_in_node=True)
    else:
        dist_print('PASSED: pre-dispatch matches the PyTorch staging path',
                   once_in_node=True)

    dist.barrier()
    dist.destroy_process_group()
    if failures and rank_idx == 0:
        sys.exit(1)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--num-processes', type=int, default=8)
    p.add_argument('--batches', type=int, nargs='+', default=[384, 2048])
    args = p.parse_args()
    torch.multiprocessing.spawn(_run, args=(args.num_processes, args),
                                nprocs=args.num_processes)
