#pragma once

#include <cstdint>
#include <type_traits>
#include <c10/macros/Macros.h>
#include <ATen/cuda/detail/OffsetCalculator.cuh>

#include <thrust/tuple.h>

// References:
// https://devblogs.nvidia.com/cuda-pro-tip-increase-performance-with-vectorized-memory-access/

namespace at { namespace native { namespace memory {

struct LoadWithoutCast {};

namespace policies {

// Assumption:
// all tensors are contiguous, that is: stride == sizeof(type) for all tensors
template<typename out_calc_t, typename loader_t>
struct unroll {
  out_calc_t output_offset_calculator;

  __device__ unroll(out_calc_t oc, loader_t l): output_offset_calculator(oc) {}
};

template <typename data_t, typename out_calc_t>
struct multi_outputs_unroll : unroll<out_calc_t, LoadWithoutCast> {
  __device__ multi_outputs_unroll(data_t data, out_calc_t oc):
    unroll<out_calc_t, LoadWithoutCast>(oc, LoadWithoutCast()) {}

  __device__ inline offset_t offsets(int linear_idx) {
    return this->output_offset_calculator.get(linear_idx);
  }
};

}  // namespace policies

}}} // namespace at::native::memory
