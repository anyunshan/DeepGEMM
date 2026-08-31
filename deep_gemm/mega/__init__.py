import torch
import types
import warnings
from typing import Tuple, Optional, Union
from ..utils.math import align

# noinspection PyBroadException
try:
    # noinspection PyProtectedMember
    import torch.distributed._symmetric_memory as symm_mem
    import torch.distributed as dist
except Exception as exception:
    print(f'Failed to load mega kernels, please check your PyTorch version: {exception}')

from .. import _C


class SymmBuffer:
    def __init__(self, group: dist.ProcessGroup,
                 num_experts: int,
                 num_max_tokens_per_rank: int, num_topk: int,
                 hidden: int, intermediate_hidden: int,
                 num_shared_experts: int = 0,
                 mma_type: str = 'fp8xfp4',
                 activation: str = 'swiglu'):
        assert activation == 'swiglu', f'Only `swiglu` activation is supported, got `{activation}`'
        self.group = group
        self.num_experts = num_experts
        self.num_max_tokens_per_rank = num_max_tokens_per_rank
        self.num_topk = num_topk
        self.hidden = hidden
        self.intermediate_hidden = intermediate_hidden

        # Allocate a symmetric buffer
        if mma_type == 'fp8xfp8':
            # SM90 (Hopper) FP8 MegaMoE: per-128-K float SF, full-pool buffers.
            # Shared experts: L1 acts alias x/x_sf; only shared L2 acts + SF are
            # real buffers (see `get_symm_buffer_size_for_sm90_fp8_mega_moe`).
            num_bytes, slice_input_buffers = _C.get_symm_buffer_size_for_sm90_fp8_mega_moe(
                group.size(), num_experts,
                num_max_tokens_per_rank, num_topk,
                hidden, intermediate_hidden,
                num_shared_experts
            )
        else:
            num_bytes, slice_input_buffers = _C.get_symm_buffer_size_for_mega_moe(
                group.size(), num_experts,
                num_max_tokens_per_rank, num_topk,
                hidden, intermediate_hidden,
                mma_type, activation,
                num_shared_experts
            )
        allocator = torch if group.size() == 1 else symm_mem
        self.buffer = allocator.empty(num_bytes, dtype=torch.int8, device='cuda')
        self.handle = (
            types.SimpleNamespace(buffer_ptrs=[self.buffer.data_ptr()])
            if group.size() == 1
            else symm_mem.rendezvous(self.buffer, group=group)
        )
        self.buffer.zero_()
        self.group.barrier()
        torch.cuda.synchronize()

        # Create input buffer views
        (self.x, self.x_sf,
         self.topk_idx, self.topk_weights,
         self.shared_l1_acts, self.shared_l1_acts_sf,
         self.shared_l2_acts, self.shared_l2_acts_sf,
         self.l1_acts, self.l1_acts_sf,
         self.l2_acts, self.l2_acts_sf) = slice_input_buffers(self.buffer)

    def destroy(self):
        self.handle = None
        self.buffer = None
        self.group = None
        self.x = None
        self.x_sf = None


def get_symm_buffer_for_mega_moe(group: dist.ProcessGroup,
                                 num_experts: int,
                                 num_max_tokens_per_rank: int, num_topk: int,
                                 hidden: int, intermediate_hidden: int,
                                 num_shared_experts: int = 0,
                                 use_fp8_dispatch: Union[bool, None] = None,
                                 mma_type: Union[str, None] = None,
                                 activation: str = 'swiglu') -> SymmBuffer:
    # Arch-dependent default MMA type: FP8xFP8 on SM90 (Hopper), FP8xFP4 on SM100
    if mma_type is None:
        mma_type = 'fp8xfp8' if torch.cuda.get_device_capability()[0] == 9 else 'fp8xfp4'

    # Align token count
    num_max_tokens_per_rank = align(num_max_tokens_per_rank, _C.get_token_alignment_for_mega_moe())

    # Backward compat: derive `mma_type` from `use_fp8_dispatch` if provided
    if use_fp8_dispatch is not None:
        assert use_fp8_dispatch == (mma_type.split('x')[0] == 'fp8')
        warnings.warn(
            f'`use_fp8_dispatch` will be deprecated in the future, please use `mma_type`',
            DeprecationWarning, stacklevel=3
        )

    return SymmBuffer(
        group, num_experts,
        num_max_tokens_per_rank, num_topk,
        hidden, intermediate_hidden,
        num_shared_experts,
        mma_type=mma_type, activation=activation
    )


def _interleave_weights(t: torch.Tensor, gran: int = 8) -> torch.Tensor:
    # [gate: 0..7, up: 0..7, gate: 8..15, up: 8..15, ...] instead of [gate | up]
    # Unsqueeze for 2D
    assert t.dim() in (2, 3)
    squeeze_group_dim = t.dim() == 2
    if squeeze_group_dim:
        t = t.unsqueeze(0)

    # Transpose
    g, n, *rest = t.shape
    half = n // 2
    gate = t[:, :half].reshape(g, half // gran, gran, *rest)
    up = t[:, half:].reshape(g, half // gran, gran, *rest)
    result = torch.empty_like(t).copy_(torch.stack([gate, up], dim=2).reshape(g, n, *rest))
    return result.squeeze(0) if squeeze_group_dim else result


def _transpose_sf_for_utccp(sf: torch.Tensor) -> torch.Tensor:
    # Unsqueeze for 2D
    assert sf.dtype == torch.int and sf.dim() in (2, 3)
    squeeze_group_dim = sf.dim() == 2
    if squeeze_group_dim:
        sf = sf.unsqueeze(0)

    # Transpose
    num_groups, mn, packed_sf_k = sf.shape
    assert mn % 128 == 0
    result = (sf.reshape(num_groups, -1, 4, 32, packed_sf_k)
                .transpose(2, 3)
                .reshape(num_groups, mn, packed_sf_k))
    result = torch.empty_like(sf).copy_(result)
    return result.squeeze(0) if squeeze_group_dim else result


def transform_weights_for_mega_moe(
    l1_weights: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    l2_weights: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    activation: str = 'swiglu'
) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
           Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]:
    assert activation == 'swiglu', f'Only `swiglu` activation is supported, got `{activation}`'
    if isinstance(l1_weights, tuple):
        # FP8: interleave gate/up for weight and SF, then transpose L1 SF for UTCCP
        l1_w = _interleave_weights(l1_weights[0])
        l1_sf = _transpose_sf_for_utccp(_interleave_weights(l1_weights[1]))
        l1_transformed = (l1_w, l1_sf)
        # L2: only transpose SF for UTCCP
        l2_transformed = (l2_weights[0], _transpose_sf_for_utccp(l2_weights[1]))
    else:
        # BF16: L1 interleave gate/up, L2 unchanged
        l1_transformed = _interleave_weights(l1_weights)
        l2_transformed = l2_weights
    return l1_transformed, l2_transformed



def fp8_fp4_mega_moe(y: torch.Tensor,
                     l1_weights: Tuple[torch.Tensor, torch.Tensor],
                     l2_weights: Tuple[torch.Tensor, torch.Tensor],
                     sym_buffer: SymmBuffer,
                     shared_l1_weights: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                     shared_l2_weights: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                     cumulative_local_expert_recv_stats: Optional[torch.Tensor] = None,
                     recipe: Tuple[int, int, int] = (1, 1, 32),
                     activation: str = 'swiglu',
                     activation_clamp: Optional[float] = None,
                     fast_math: bool = True):
    _C.fp8_fp4_mega_moe(
        y,
        l1_weights, l2_weights,
        shared_l1_weights, shared_l2_weights,
        cumulative_local_expert_recv_stats,
        sym_buffer.buffer,
        sym_buffer.handle.buffer_ptrs, sym_buffer.group.rank(),
        sym_buffer.num_max_tokens_per_rank,
        sym_buffer.num_experts, sym_buffer.num_topk,
        recipe,
        activation, activation_clamp,
        fast_math
    )

def bf16_mega_moe(y: torch.Tensor,
                  l1_weights: torch.Tensor,
                  l2_weights: torch.Tensor,
                  sym_buffer: SymmBuffer,
                  shared_l1_weights: Optional[torch.Tensor] = None,
                  shared_l2_weights: Optional[torch.Tensor] = None,
                  cumulative_local_expert_recv_stats: Optional[torch.Tensor] = None,
                  activation: str = 'swiglu',
                  activation_clamp: Optional[float] = None,
                  fast_math: bool = True):
    _C.bf16_mega_moe(
        y,
        l1_weights,
        l2_weights,
        shared_l1_weights,
        shared_l2_weights,
        cumulative_local_expert_recv_stats,
        sym_buffer.buffer,
        sym_buffer.handle.buffer_ptrs,
        sym_buffer.group.rank(),
        sym_buffer.num_max_tokens_per_rank,
        sym_buffer.num_experts,
        sym_buffer.num_topk,
        activation, activation_clamp,
        fast_math
    )


def transform_weights_for_mega_moe_sm90(
    l1_weights: Tuple[torch.Tensor, torch.Tensor],
    l2_weights: Tuple[torch.Tensor, torch.Tensor],
    shared_l1_weights: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    shared_l2_weights: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
):
    """SM90 (Hopper) variant of `transform_weights_for_mega_moe`.

    SM90 has no TMEM / UTCCP path, so the SF tensors are consumed directly by
    WGMMA promote and don't need the 4x32 transpose. With block (128, 128)
    weight quantization, weight SFs are read by the math warpgroup directly
    from global memory in their natural ``(E, N/128, K/128)`` MN-major layout
    and require no transformation. Only L1's gate/up FP8 weight interleave is
    preserved (granularity 8, matching the WGMMA accumulator group width).
    Shared L1 weights (2D) get the same gate/up interleave; shared L2 and all
    SFs pass through unchanged.
    """
    l1_fp8, l1_sf = l1_weights
    if shared_l1_weights is None:
        return (_interleave_weights(l1_fp8), l1_sf), l2_weights
    s1_fp8, s1_sf = shared_l1_weights
    return ((_interleave_weights(l1_fp8), l1_sf), l2_weights,
            (_interleave_weights(s1_fp8), s1_sf), shared_l2_weights)


def mega_moe_pre_dispatch_sm90(x: torch.Tensor,
                               topk_idx: torch.Tensor,
                               topk_weights: torch.Tensor,
                               buf_x: torch.Tensor,
                               buf_x_sf: torch.Tensor,
                               buf_topk_idx: torch.Tensor,
                               buf_topk_weights: torch.Tensor,
                               num_tokens: int,
                               group_size: int = 128,
                               routed_scaling_factor: float = 1.0) -> None:
    """Stage BF16 activations + routing into the symmetric buffer for SM90 MegaMoE.

    Quantizes ``x`` to FP8 e4m3 with per-token per-``group_size``-K float scale
    factors, writes it (plus ``topk_idx`` / ``topk_weights``) directly into the
    buffer views returned by ``get_symm_buffer_for_mega_moe``, and folds
    ``routed_scaling_factor`` into the stored weights. Slots between
    ``num_tokens`` and the buffer capacity are marked inactive
    (``topk_idx = -1``, weight 0), which is what the fused kernel expects when a
    batch is shorter than the allocated maximum.

    Doing this in one kernel matters: the PyTorch equivalent (cast, copy x, copy
    sf, scale+copy weights) costs four HBM round trips before the fused kernel
    even starts.

    ``topk_idx`` is int32 here (the buffer stores int64; the kernel converts).
    """
    _C.mega_moe_pre_dispatch_sm90(
        x, topk_idx, topk_weights,
        buf_x, buf_x_sf, buf_topk_idx, buf_topk_weights,
        num_tokens, group_size, float(routed_scaling_factor),
    )


def fp8_mega_moe(y: torch.Tensor,
                 l1_weights: Tuple[torch.Tensor, torch.Tensor],
                 l2_weights: Tuple[torch.Tensor, torch.Tensor],
                 sym_buffer: SymmBuffer,
                 shared_l1_weights: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                 shared_l2_weights: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                 cumulative_local_expert_recv_stats: Optional[torch.Tensor] = None,
                 recipe: Tuple[int, int, int] = (128, 128, 128),
                 activation: str = 'swiglu',
                 activation_clamp: Optional[float] = None,
                 fast_math: bool = True):
    """SM90 (Hopper) MegaMoE entry point.

    Expects FP8 e4m3 weights and block-(128, 128) float scale factors. Backed by
    the single N-split kernel (BLOCK_M=64, BLOCK_N=256): two math warpgroups
    split each tile into two 128-wide halves, share one A-tile load, and
    quantize the L2-input activations at per-128 K (matches the standard DeepEP
    runner). Shared experts (optional, 2D FP8 weight pairs) run as fused dense
    phases; their output enters combine with weight 1.0.
    """
    _C.fp8_mega_moe(
        y,
        l1_weights, l2_weights,
        shared_l1_weights, shared_l2_weights,
        cumulative_local_expert_recv_stats,
        sym_buffer.buffer,
        sym_buffer.handle.buffer_ptrs, sym_buffer.group.rank(),
        sym_buffer.num_max_tokens_per_rank,
        sym_buffer.num_experts, sym_buffer.num_topk,
        recipe,
        activation, activation_clamp,
        fast_math
    )
