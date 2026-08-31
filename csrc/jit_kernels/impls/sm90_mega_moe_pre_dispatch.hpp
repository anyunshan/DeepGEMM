#pragma once

#include <torch/python.h>
#include <string>

#include "../../jit/compiler.hpp"
#include "../../jit/kernel_runtime.hpp"
#include "../../utils/exception.hpp"
#include "../../utils/format.hpp"
#include "../../utils/math.hpp"
#include "runtime_utils.hpp"

namespace deep_gemm {

// SM90 MegaMoE pre-dispatch: quantize BF16 activations to FP8 with per-128-K
// float scale factors, write them straight into the symmetric buffer together
// with the routing decisions, and fold `routed_scaling_factor` into the topk
// weights. One kernel instead of the four HBM round trips a PyTorch equivalent
// costs (cast -> copy x -> copy sf -> scale+copy weights).
//
// The padding tail matters: the fused kernel walks `num_max_tokens_per_rank`
// worth of topk slots, so slots past `num_tokens` must read as inactive
// (`topk_idx = -1`, weight 0). Blocks beyond `num_tokens` do exactly that.
class SM90MegaMoEPreDispatchRuntime final : public LaunchRuntime<SM90MegaMoEPreDispatchRuntime> {
public:
    struct Args {
        // Templated arguments
        int group_size;
        bool use_pdl;

        // Runtime arguments
        const void* x;
        const int* topk_idx;
        const float* topk_weights;
        void* buf_x;
        float* buf_x_sf;
        int64_t* buf_topk_idx;
        float* buf_topk_weights;
        int num_tokens;
        int padded_max;
        int hidden;
        int num_groups;
        int top_k;
        float routed_scaling_factor;

        LaunchArgs launch_args;
    };

    static std::string generate_impl(const Args& args) {
        return fmt::format(R"(
// JIT cache key: sm90_mega_moe_pre_dispatch_v1
#include <deep_gemm/impls/sm90_mega_moe_pre_dispatch.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sm90_mega_moe_pre_dispatch_kernel<
        {}, {}
    >);
}};
)",
    args.group_size,
    args.use_pdl ? "true" : "false");
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        DG_CUDA_UNIFIED_CHECK(launch_kernel(kernel, config,
            args.x,
            args.topk_idx,
            args.topk_weights,
            args.buf_x,
            args.buf_x_sf,
            args.buf_topk_idx,
            args.buf_topk_weights,
            static_cast<uint32_t>(args.num_tokens),
            static_cast<uint32_t>(args.padded_max),
            static_cast<uint32_t>(args.hidden),
            static_cast<uint32_t>(args.num_groups),
            static_cast<uint32_t>(args.top_k),
            args.routed_scaling_factor));
    }
};

static void sm90_mega_moe_pre_dispatch(
    const torch::Tensor& x,
    const torch::Tensor& topk_idx,
    const torch::Tensor& topk_weights,
    const torch::Tensor& buf_x,
    const torch::Tensor& buf_x_sf,
    const torch::Tensor& buf_topk_idx,
    const torch::Tensor& buf_topk_weights,
    const int& num_tokens,
    const int& group_size,
    const float& routed_scaling_factor) {
    const auto hidden = static_cast<int>(x.size(1));
    const auto top_k = static_cast<int>(topk_idx.size(1));
    const auto padded_max = static_cast<int>(buf_topk_idx.size(0));
    const auto num_groups = hidden / group_size;

    // One thread per 8 BF16 elements of a token row, so the row must fit a
    // single block and divide the 16-byte vector load.
    constexpr int kVecElems = 8;
    const auto num_threads = hidden / kVecElems;
    DG_HOST_ASSERT(group_size == 128);
    DG_HOST_ASSERT(hidden % (group_size * 1) == 0);
    DG_HOST_ASSERT(hidden % kVecElems == 0);
    DG_HOST_ASSERT(num_threads > 0 and num_threads <= 1024);
    DG_HOST_ASSERT(top_k <= num_threads);

    // Padding blocks: one block covers `num_threads` topk slots
    const auto num_pad_slots = (padded_max - num_tokens) * top_k;
    const auto num_pad_blocks = num_pad_slots > 0
                                    ? ceil_div(num_pad_slots, num_threads)
                                    : 0;
    const auto num_blocks = num_tokens + num_pad_blocks;

    // A zero-token launch still has to blank the padding tail; if there is
    // nothing at all to do, skip the launch (grid dim 0 is illegal).
    if (num_blocks == 0)
        return;

    const auto& args = SM90MegaMoEPreDispatchRuntime::Args{
        .group_size = group_size,
        .use_pdl = device_runtime->get_pdl(),
        .x = x.data_ptr(),
        .topk_idx = topk_idx.data_ptr<int>(),
        .topk_weights = topk_weights.data_ptr<float>(),
        .buf_x = buf_x.data_ptr(),
        .buf_x_sf = buf_x_sf.data_ptr<float>(),
        .buf_topk_idx = buf_topk_idx.data_ptr<int64_t>(),
        .buf_topk_weights = buf_topk_weights.data_ptr<float>(),
        .num_tokens = num_tokens,
        .padded_max = padded_max,
        .hidden = hidden,
        .num_groups = num_groups,
        .top_k = top_k,
        .routed_scaling_factor = routed_scaling_factor,
        .launch_args = LaunchArgs(num_blocks, num_threads)
    };
    const auto& code = SM90MegaMoEPreDispatchRuntime::generate(args);
    const auto& runtime = compiler->build("sm90_mega_moe_pre_dispatch", code);
    SM90MegaMoEPreDispatchRuntime::launch(runtime, args);
}

} // namespace deep_gemm
