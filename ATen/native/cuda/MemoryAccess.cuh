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
  data_t data;
  int remaining;

  __device__ multi_outputs_unroll(data_t data, int remaining, out_calc_t oc):
    unroll<out_calc_t, LoadWithoutCast>(oc, LoadWithoutCast()),
    data(data), remaining(remaining) {}

  __device__ inline bool check_inbounds(int thread_work_elem) {
    return ((threadIdx.x  + thread_work_elem*num_threads) < this->remaining);
  }

  __device__ inline void load(std::tuple<float, float> *args, int idx) {
    int thread_idx = threadIdx.x;
    #pragma unroll
    for (int i = 0; i < 1; i++) {
      if (thread_idx >= remaining) {
        return;
      }
      int linear_idx = thread_idx + block_work_size * idx;
      std::get<0>(args[i]) = *(reinterpret_cast<float *>(data[2]) + linear_idx);
      std::get<1>(args[i]) = *(reinterpret_cast<float *>(data[3]) + linear_idx);
      thread_idx += num_threads;
    }
  }

  __device__ inline void store(thrust::tuple<float, float> *from, int idx) {
    int thread_idx = threadIdx.x;
    #pragma unroll
    for (int i = 0; i < 1; i++) {
      if (thread_idx >= this->remaining) {
        return;
      }
      int linear_idx = thread_idx + block_work_size * idx;
      auto offsets = this->output_offset_calculator.get(linear_idx);
      *(reinterpret_cast<float *>(this->data[0]) + offsets[0]) = thrust::get<0>(from[i]);
      *(reinterpret_cast<float *>(this->data[1]) + offsets[1]) = thrust::get<1>(from[i]);
      thread_idx += num_threads;
    }
  }
};

}  // namespace policies

}}} // namespace at::native::memory
