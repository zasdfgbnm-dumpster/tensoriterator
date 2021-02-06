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
    using arg_t = std::tuple_element_t<arg_index, args_t>;
    // `data` hold the data_ptr for tensors [output, input0, input1, ...], so we
    // need a +1 offset to get the input
    std::get<arg_index>(args[j]) = loader.template load<arg_t>(self.data[arg_index + num_outputs], offset[arg_index], arg_index);
  }
};

template <int current>
struct multi_outputs_store_helper {
  C10_HOST_DEVICE static void apply(
      at::detail::Array<char*, 4> data,
      at::detail::Array<uint32_t, 2> offsets,
      thrust::tuple<float, float> ret) {
    float *to = reinterpret_cast<float *>(data[current]) + offsets[current];
    *to = thrust::get<current>(ret);
  }
};

}  // namespace detail

struct LoadWithoutCast {
  template<typename scalar_t>
  __device__ scalar_t load(char *base_ptr, uint32_t offset, int arg) {
    return *(reinterpret_cast<scalar_t *>(base_ptr) + offset);
  }
};

struct StoreWithoutCast {
  template<typename scalar_t>
  __device__ void store(scalar_t value, char *base_ptr, uint32_t offset) {
    *(reinterpret_cast<scalar_t *>(base_ptr) + offset) = value;
  }
};


// aligned vector generates vectorized load/store on CUDA
template<typename scalar_t, int vec_size>
struct alignas(sizeof(scalar_t) * vec_size) aligned_vector {
  scalar_t val[vec_size];
};

namespace policies {

// Assumption:
// all tensors are contiguous, that is: stride == sizeof(type) for all tensors
template<typename data_t, typename inp_calc_t, typename out_calc_t, typename loader_t, typename storer_t, int num_outputs = 1>
struct unroll {

  data_t data;
  int remaining;
  inp_calc_t input_offset_calculator;
  out_calc_t output_offset_calculator;
  loader_t loader;
  storer_t storer;

  __device__ unroll(data_t data, int remaining, inp_calc_t ic, out_calc_t oc, loader_t l, storer_t s):
    data(data), remaining(remaining), input_offset_calculator(ic), output_offset_calculator(oc), loader(l), storer(s) {}

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

  template<typename scalar_t>
  __device__ inline void store(scalar_t *from, int idx) {}
};

template <typename data_t, typename inp_calc_t, typename out_calc_t, int num_outputs>
struct multi_outputs_unroll : unroll<data_t, inp_calc_t, out_calc_t, LoadWithoutCast, StoreWithoutCast, num_outputs> {

  __device__ multi_outputs_unroll(data_t data, int remaining, inp_calc_t ic, out_calc_t oc):
    unroll<data_t, inp_calc_t, out_calc_t, LoadWithoutCast, StoreWithoutCast, num_outputs>(data, remaining, ic, oc, LoadWithoutCast(), StoreWithoutCast()) {}

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
      detail::multi_outputs_store_helper<0>::apply(this->data, offsets, from[i]);
      detail::multi_outputs_store_helper<1>::apply(this->data, offsets, from[i]);
      thread_idx += num_threads;
    }
  }
};

}  // namespace policies

}}} // namespace at::native::memory
