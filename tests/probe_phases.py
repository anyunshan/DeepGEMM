"""Quantify where the fused shared-expert path spends its time.

No kernel instrumentation: each phase cost is obtained by differencing
configurations that differ in exactly one phase.

  A. mega(ns=0)                     routed only  = dispatch + routed GEMM + combine
  B. mega(ns=1)                     routed + shared fused
  C. shared L1 GEMM alone           standalone dense GEMM (single GPU, no comm)
  D. shared L2 GEMM alone           standalone dense GEMM (single GPU, no comm)
  E. mega(ns=0) with 1 token        dispatch/barrier floor (GEMM work ~ 0)

Derived:
  fusion_cost   = B - A             what fusing shared actually adds
  shared_alone  = C + D             what the same math costs unfused
  hidden        = shared_alone - fusion_cost    how much got overlapped
  dispatch_floor= E                 fixed cross-rank cost, the size of the
                                    window shared L1 can hide inside
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
GLM = dict(hidden=6144, intermediate_hidden=2048, num_experts=256, num_topk=8)


def _make_inputs(num_tokens, num_max, ns, rank_idx, num_ranks):
    H, IH = GLM['hidden'], GLM['intermediate_hidden']
    E, K = GLM['num_experts'], GLM['num_topk']
    Epr = E // num_ranks
    shared_ih = IH * ns if ns else IH

    torch.manual_seed(1000 + rank_idx)
    x_bf = torch.randn(max(num_tokens, 1), H, dtype=torch.bfloat16, device='cuda')
    x_fp8, x_sf = per_token_cast_to_fp8(x_bf, use_ue8m0=False, gran_k=128,
                                        use_packed_ue8m0=False)
    l1_bf = torch.randn(Epr, 2 * IH, H, dtype=torch.bfloat16, device='cuda') * 0.05
    l2_bf = torch.randn(Epr, H, IH, dtype=torch.bfloat16, device='cuda') * 0.05
    s1_bf = torch.randn(2 * shared_ih, H, dtype=torch.bfloat16, device='cuda') * 0.05
    s2_bf = torch.randn(H, shared_ih, dtype=torch.bfloat16, device='cuda') * 0.05

    scores = torch.randn(max(num_tokens, 1), E, dtype=torch.float, device='cuda')
    topk_w, topk_idx = torch.topk(scores, K, dim=-1, largest=True, sorted=False)

    return dict(
        x_fp8=x_fp8, x_sf=x_sf, topk_idx=topk_idx, topk_w=topk_w,
        l1_w=_quantize_grouped_fp8_block_128_128(l1_bf),
        l2_w=_quantize_grouped_fp8_block_128_128(l2_bf),
        s1_w=_quantize_2d_fp8_block_128_128(s1_bf),
        s2_w=_quantize_2d_fp8_block_128_128(s2_bf),
        num_experts_per_rank=Epr, shared_ih=shared_ih,
    )


def _time_mega(num_tokens, num_max, ns, inp, group, num_tests):
    """Time one fp8_mega_moe launch (ns=0 routed-only or ns=1 fused)."""
    H, IH = GLM['hidden'], GLM['intermediate_hidden']
    E, K = GLM['num_experts'], GLM['num_topk']

    if ns:
        t_l1, t_l2, t_s1, t_s2 = deep_gemm.transform_weights_for_mega_moe_sm90(
            inp['l1_w'], inp['l2_w'], inp['s1_w'], inp['s2_w'])
    else:
        t_l1, t_l2 = deep_gemm.transform_weights_for_mega_moe_sm90(
            inp['l1_w'], inp['l2_w'])
        t_s1 = t_s2 = None

    buf = deep_gemm.get_symm_buffer_for_mega_moe(
        group, E, num_max, K, H, IH, num_shared_experts=ns)
    cum = torch.zeros(inp['num_experts_per_rank'], dtype=torch.int, device='cuda')
    y = torch.empty(max(num_tokens, 1), H, dtype=torch.bfloat16, device='cuda')

    def run():
        if num_tokens:
            buf.x[:num_tokens].copy_(inp['x_fp8'][:num_tokens])
            buf.x_sf[:num_tokens].copy_(inp['x_sf'][:num_tokens])
            buf.topk_idx[:num_tokens].copy_(inp['topk_idx'][:num_tokens])
            buf.topk_weights[:num_tokens].copy_(inp['topk_w'][:num_tokens])
        deep_gemm.fp8_mega_moe(
            y[:num_tokens] if num_tokens else y[:0], t_l1, t_l2, buf,
            shared_l1_weights=t_s1, shared_l2_weights=t_s2,
            cumulative_local_expert_recv_stats=cum,
            recipe=(128, 128, 128), activation='swiglu',
            activation_clamp=10.0, fast_math=True)

    run()
    torch.cuda.synchronize()
    dist.barrier()
    t = bench(run, num_warmups=5, num_tests=num_tests)
    dist.barrier()
    buf.destroy()
    return t * 1e6


def _time_shared_gemms(num_tokens, inp, num_tests):
    """Time the two standalone dense shared GEMMs (no comm, single GPU)."""
    H = GLM['hidden']
    shared_ih = inp['shared_ih']
    n = max(num_tokens, 1)

    l1_out = torch.empty(n, shared_ih * 2, dtype=torch.bfloat16, device='cuda')
    act_fp8 = torch.zeros(n, shared_ih, dtype=torch.float8_e4m3fn, device='cuda')
    act_sf = torch.ones(n, shared_ih // 128, dtype=torch.float32, device='cuda')
    y_shared = torch.empty(n, H, dtype=torch.bfloat16, device='cuda')

    x_pair = (inp['x_fp8'][:n], inp['x_sf'][:n])

    def run_l1():
        deep_gemm.fp8_gemm_nt(x_pair, inp['s1_w'], l1_out,
                              recipe=_DENSE_RECIPE, disable_ue8m0_cast=True)

    def run_l2():
        deep_gemm.fp8_gemm_nt((act_fp8, act_sf), inp['s2_w'], y_shared,
                              recipe=_DENSE_RECIPE, disable_ue8m0_cast=True)

    run_l1(); run_l2()
    torch.cuda.synchronize()
    t1 = bench(run_l1, num_warmups=5, num_tests=num_tests) * 1e6
    t2 = bench(run_l2, num_warmups=5, num_tests=num_tests) * 1e6
    return t1, t2


def test(local_rank, num_local_ranks, args):
    rank_idx, num_ranks, group = init_dist(local_rank, num_local_ranks)
    if get_arch_major() != 9:
        dist_print(f'[SKIP] requires SM90', once_in_node=True)
        dist.destroy_process_group()
        return

    dist_print(f'Phase probe: GLM-5.2 shape, {num_ranks} ranks, '
               f'{args.num_tests} tests each', once_in_node=True)
    dist_print('', once_in_node=True)
    dist_print(f'{"tokens":>7} {"routed":>9} {"fused":>9} {"fusion+":>9} '
               f'{"sL1":>7} {"sL2":>7} {"sTotal":>8} {"hidden":>8} {"hid%":>6}',
               once_in_node=True)

    # Dispatch floor: 1 token, ns=0 -> almost pure cross-rank cost
    inp_floor = _make_inputs(1, args.token_align, 0, rank_idx, num_ranks)
    t_floor = _time_mega(1, args.token_align, 0, inp_floor, group, args.num_tests)

    for tokens in args.batches:
        num_max = tokens
        inp = _make_inputs(tokens, num_max, 1, rank_idx, num_ranks)
        t_routed = _time_mega(tokens, num_max, 0, inp, group, args.num_tests)
        t_fused = _time_mega(tokens, num_max, 1, inp, group, args.num_tests)
        t_s1, t_s2 = _time_shared_gemms(tokens, inp, args.num_tests)

        fusion_cost = t_fused - t_routed
        shared_alone = t_s1 + t_s2
        hidden = shared_alone - fusion_cost
        hid_pct = 100.0 * hidden / shared_alone if shared_alone > 0 else 0.0

        dist_print(f'{tokens:>7} {t_routed:>8.0f}u {t_fused:>8.0f}u '
                   f'{fusion_cost:>8.0f}u {t_s1:>6.0f}u {t_s2:>6.0f}u '
                   f'{shared_alone:>7.0f}u {hidden:>7.0f}u {hid_pct:>5.0f}%',
                   once_in_node=True)

    dist_print('', once_in_node=True)
    dist_print(f'Dispatch floor (1 token, ns=0): {t_floor:.0f} us', once_in_node=True)
    dist_print('  = fixed cross-rank cost; shared L1 can only hide inside a window '
               'of about this size', once_in_node=True)
    dist_print('', once_in_node=True)
    dist_print('fusion+ = fused - routed (what fusion adds)', once_in_node=True)
    dist_print('sTotal  = shared L1 + L2 measured standalone', once_in_node=True)
    dist_print('hidden  = sTotal - fusion+ (overlap achieved); hid% of sTotal',
               once_in_node=True)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--num-processes', type=int, default=8)
    p.add_argument('--num-tests', type=int, default=20)
    p.add_argument('--batches', type=int, nargs='+', default=[128, 512, 2048, 8192])
    p.add_argument('--token-align', type=int, default=384)
    args = p.parse_args()
    torch.multiprocessing.spawn(test, args=(args.num_processes, args),
                                nprocs=args.num_processes)

