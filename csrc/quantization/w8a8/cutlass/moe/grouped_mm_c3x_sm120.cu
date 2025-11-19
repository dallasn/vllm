// SM120 (GeForce Blackwell) MoE Grouped GEMM kernels - FP8 standard scaling
// Optimized for RTX 5090 with native SM120 schedules
// Requires CUTLASS v4.3.0+ and CUDA 12.8+
//
// Key optimizations:
// - Native KernelPtrArrayTmaWarpSpecializedPingpongSm120 schedules
// - Larger tiles (256x128x128 default) for better SM utilization  
// - 2SM cooperation for default config
// - Dynamic tile selection based on problem size

#include <cudaTypedefs.h>

#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>

#include "cutlass/cutlass.h"
#include "grouped_mm_c3x.cuh"

using namespace cute;

namespace {

template <typename InType, typename OutType,
          template <typename, typename, typename> typename Epilogue>
struct sm120_fp8_config_default {
  // SM120 (GeForce Blackwell) default configuration with native schedules
  // Uses 2SM cooperation for better throughput
  static_assert(std::is_same<InType, cutlass::float_e4m3_t>());
  
  // Native SM120 pingpong schedule for FP8
  using KernelSchedule =
      cutlass::gemm::KernelPtrArrayTmaWarpSpecializedPingpongSm120;
  using EpilogueSchedule =
      cutlass::epilogue::PtrArrayTmaWarpSpecializedPingpong;
  
  // Larger tile with 2SM cooperation for better utilization
  using TileShape = cute::Shape<cute::_256, cute::_128, cute::_128>;
  using ClusterShape = cute::Shape<cute::_2, cute::_1, cute::_1>;
  using ArchTag = cutlass::arch::Sm120;

  using Cutlass3xGemm =
      cutlass_3x_group_gemm<InType, OutType, ArchTag, Epilogue, TileShape,
                            ClusterShape, KernelSchedule, EpilogueSchedule>;
};

template <typename InType, typename OutType,
          template <typename, typename, typename> typename Epilogue>
struct sm120_fp8_config_M4 {
  // SM120 small M configuration for tiny expert batches
  static_assert(std::is_same<InType, cutlass::float_e4m3_t>());
  
  // Native SM120 schedule even for small M
  using KernelSchedule =
      cutlass::gemm::KernelPtrArrayTmaWarpSpecializedPingpongSm120;
  using EpilogueSchedule =
      cutlass::epilogue::PtrArrayTmaWarpSpecializedPingpong;
  
  using TileShape = cute::Shape<cute::_128, cute::_16, cute::_128>;
  using ClusterShape = cute::Shape<cute::_1, cute::_1, cute::_1>;
  using ArchTag = cutlass::arch::Sm120;

  using Cutlass3xGemm =
      cutlass_3x_group_gemm<InType, OutType, ArchTag, Epilogue, TileShape,
                            ClusterShape, KernelSchedule, EpilogueSchedule,
                            true>;
};

template <typename InType, typename OutType,
          template <typename, typename, typename> typename Epilogue>
struct sm120_fp8_config_M16 {
  // SM120 medium M configuration with native schedule
  static_assert(std::is_same<InType, cutlass::float_e4m3_t>());
  
  // Native SM120 schedule
  using KernelSchedule =
      cutlass::gemm::KernelPtrArrayTmaWarpSpecializedPingpongSm120;
  using EpilogueSchedule =
      cutlass::epilogue::PtrArrayTmaWarpSpecializedPingpong;
  
  // Moderate tile size for medium M
  using TileShape = cute::Shape<cute::_128, cute::_128, cute::_128>;
  using ClusterShape = cute::Shape<cute::_1, cute::_1, cute::_1>;
  using ArchTag = cutlass::arch::Sm120;

  using Cutlass3xGemm =
      cutlass_3x_group_gemm<InType, OutType, ArchTag, Epilogue, TileShape,
                            ClusterShape, KernelSchedule, EpilogueSchedule>;
};

}  // namespace

template <typename InType, typename OutType,
          template <typename, typename, typename> typename Epilogue>
void cutlass_moe_mm_sm120_dispatcher(
    torch::Tensor& out_tensors, torch::Tensor const& a_tensors,
    torch::Tensor const& b_tensors, torch::Tensor const& a_scales,
    torch::Tensor const& b_scales, torch::Tensor const& expert_offsets,
    torch::Tensor const& problem_sizes, torch::Tensor const& a_strides,
    torch::Tensor const& b_strides, torch::Tensor const& c_strides,
    bool per_act_token, bool per_out_ch, int64_t max_m) {
  
  // Dispatch based on M size for optimal performance
  if (max_m <= 4) {
    return sm120_fp8_config_M4<InType, OutType, Epilogue>::Cutlass3xGemm::run(
        out_tensors, a_tensors, b_tensors, a_scales, b_scales, expert_offsets,
        problem_sizes, a_strides, b_strides, c_strides, per_act_token,
        per_out_ch);
  } else if (max_m <= 16) {
    return sm120_fp8_config_M16<InType, OutType, Epilogue>::Cutlass3xGemm::run(
        out_tensors, a_tensors, b_tensors, a_scales, b_scales, expert_offsets,
        problem_sizes, a_strides, b_strides, c_strides, per_act_token,
        per_out_ch);
  } else {
    return sm120_fp8_config_default<InType, OutType, Epilogue>::Cutlass3xGemm::run(
        out_tensors, a_tensors, b_tensors, a_scales, b_scales, expert_offsets,
        problem_sizes, a_strides, b_strides, c_strides, per_act_token,
        per_out_ch);
  }
}

void cutlass_moe_mm_sm120(
    torch::Tensor& out_tensors, torch::Tensor const& a_tensors,
    torch::Tensor const& b_tensors, torch::Tensor const& a_scales,
    torch::Tensor const& b_scales, torch::Tensor const& expert_offsets,
    torch::Tensor const& problem_sizes, torch::Tensor const& a_strides,
    torch::Tensor const& b_strides, torch::Tensor const& c_strides,
    bool per_act_token, bool per_out_ch) {
  
  // Get max M size for dispatching
  int64_t num_experts = expert_offsets.numel();
  int64_t max_m = 0;
  auto problem_sizes_ptr = problem_sizes.data_ptr<int>();
  for (int64_t i = 0; i < num_experts; ++i) {
    max_m = std::max(max_m, static_cast<int64_t>(problem_sizes_ptr[i * 3]));
  }

  TORCH_CHECK(a_tensors.dtype() == torch::kFloat8_e4m3fn,
              "SM120 MoE only supports FP8 E4M3 inputs currently");
  
  if (out_tensors.dtype() == torch::kFloat32) {
    return cutlass_moe_mm_sm120_dispatcher<cutlass::float_e4m3_t, float,
                                            vllm::ScaledEpilogue>(
        out_tensors, a_tensors, b_tensors, a_scales, b_scales,
        expert_offsets, problem_sizes, a_strides, b_strides, c_strides,
        per_act_token, per_out_ch, max_m);
  } else if (out_tensors.dtype() == torch::kBFloat16) {
    return cutlass_moe_mm_sm120_dispatcher<cutlass::float_e4m3_t,
                                            cutlass::bfloat16_t,
                                            vllm::ScaledEpilogue>(
        out_tensors, a_tensors, b_tensors, a_scales, b_scales,
        expert_offsets, problem_sizes, a_strides, b_strides, c_strides,
        per_act_token, per_out_ch, max_m);
  } else if (out_tensors.dtype() == torch::kFloat16) {
    return cutlass_moe_mm_sm120_dispatcher<cutlass::float_e4m3_t,
                                            cutlass::half_t,
                                            vllm::ScaledEpilogue>(
        out_tensors, a_tensors, b_tensors, a_scales, b_scales,
        expert_offsets, problem_sizes, a_strides, b_strides, c_strides,
        per_act_token, per_out_ch, max_m);
  } else {
    TORCH_CHECK(false, "SM120 MoE unsupported output dtype: ", out_tensors.dtype());
  }
}
