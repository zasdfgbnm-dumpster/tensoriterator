#pragma once

#include <cstdint>
#include <type_traits>
#include <c10/macros/Macros.h>
#include <ATen/cuda/detail/OffsetCalculator.cuh>

#include <thrust/tuple.h>

// References:
// https://devblogs.nvidia.com/cuda-pro-tip-increase-performance-with-vectorized-memory-access/

namespace at { namespace native { namespace memory {

namespace detail {

template<int arg_index>
struct unroll_load_helper {
  template <typename args_t, typename policy_t, typename offset_t, typename loader_t>
  static __device__ void apply(policy_t &self, args_t *args, offset_t offset, loader_t loader, int j, int num_outputs) {
    // `data` hold the data_ptr for tensors [output, input0, input1, ...], so we
    // need a +1 offset to get the input
    std::get<arg_index>(args[j]) = *(reinterpret_cast<float *>(self.data[arg_index + 2]) + offset[arg_index]);
  }
};

}  // namespace detail

struct LoadWithoutCast {};

namespace policies {

// Assumption:
// all tensors are contiguous, that is: stride == sizeof(type) for all tensors
template<typename data_t, typename inp_calc_t, typename out_calc_t, typename loader_t, int num_outputs = 1>
struct unroll {

  data_t data;
  int remaining;
  inp_calc_t input_offset_calculator;
  out_calc_t output_offset_calculator;
  loader_t loader;

  __device__ unroll(data_t data, int remaining, inp_calc_t ic, out_calc_t oc, loader_t l):
    data(data), remaining(remaining), input_offset_calculator(ic), output_offset_calculator(oc), loader(l) {}

  __device__ inline bool check_inbounds(int thread_work_elem) {
    return ((threadIdx.x  + thread_work_elem*num_threads) < remaining);
  }

  template<typename args_t>
  __device__ inline void load(args_t *args, int idx) {
    int thread_idx = threadIdx.x;
    #pragma unroll
    for (int i = 0; i < thread_work_size; i++) {
      if (thread_idx >= remaining) {
        return;
      }
      int linear_idx = thread_idx + block_work_size * idx;
      auto offset = input_offset_calculator.get(linear_idx);
      detail::unroll_load_helper<0>::apply(*this, args, offset, loader, i, num_outputs);
      detail::unroll_load_helper<1>::apply(*this, args, offset, loader, i, num_outputs);
      thread_idx += num_threads;
    }
  }
};

template <typename data_t, typename inp_calc_t, typename out_calc_t, int num_outputs>
struct multi_outputs_unroll : unroll<data_t, inp_calc_t, out_calc_t, LoadWithoutCast, num_outputs> {

  __device__ multi_outputs_unroll(data_t data, int remaining, inp_calc_t ic, out_calc_t oc):
    unroll<data_t, inp_calc_t, out_calc_t, LoadWithoutCast, num_outputs>(data, remaining, ic, oc, LoadWithoutCast()) {}

  template <typename return_t>
  __device__ inline void store(return_t *from, int idx) {
    int thread_idx = threadIdx.x;
    #pragma unroll
    for (int i = 0; i < thread_work_size; i++) {
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
