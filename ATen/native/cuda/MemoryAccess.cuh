#pragma once

#include <cstdint>
#include <type_traits>
#include <c10/util/TypeCast.h>
#include <c10/macros/Macros.h>
#include <ATen/core/Array.h>
#include <ATen/detail/FunctionTraits.h>
#include <ATen/cuda/detail/OffsetCalculator.cuh>

#include <thrust/tuple.h>

// References:
// https://devblogs.nvidia.com/cuda-pro-tip-increase-performance-with-vectorized-memory-access/

namespace at { namespace native { namespace memory {

namespace detail {

// What does the `static_unroll` do?
//
// We want to do something like:
//
//    using args_t = typename traits::ArgsTuple;
//    args_t args;
//    #pragma unroll
//    for (int i = 0; i < traits::arity; i++) {
//      std::get<i>(args) = ....
//    }
//
// but unfortunately the above code does not work because
// the template argument has to be a compile time constant
// so `static_unroll` is created to simulate `#pragma unroll`
// using template metaprogramming.

template<template<int i> typename func, int end, int current=0>
struct static_unroll {
  template<typename... Args>
  static inline C10_HOST_DEVICE void with_args(Args&&... args) {
    func<current>::apply(std::forward<Args>(args)...);
    static_unroll<func, end, current+1>::with_args(args...);
  }
};

template<template<int i> typename func, int end>
struct static_unroll<func, end, end> {
  template<typename... Args>
  static inline C10_HOST_DEVICE void with_args(Args... args) {}
};

// helper structs to be used with static_unroll to load arguments
// one by one

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
  template<int ntensors, int num_outputs, typename ...Args>
  C10_HOST_DEVICE static void apply(
      at::detail::Array<char*, ntensors> data,
      at::detail::Array<uint32_t, num_outputs> offsets,
      thrust::tuple<Args...> ret) {
    using T = typename thrust::tuple_element<current, thrust::tuple<Args...>>::type;
    T *to = reinterpret_cast<T *>(data[current]) + offsets[current];
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
    constexpr int arity = std::tuple_size<args_t>::value;
    int thread_idx = threadIdx.x;
    #pragma unroll
    for (int i = 0; i < thread_work_size; i++) {
      if (thread_idx >= remaining) {
        return;
      }
      int linear_idx = thread_idx + block_work_size * idx;
      auto offset = input_offset_calculator.get(linear_idx);
      detail::static_unroll<detail::unroll_load_helper, arity>::with_args(*this, args, offset, loader, i, num_outputs);
      thread_idx += num_threads;
    }
  }

  template<typename scalar_t>
  __device__ inline void store(scalar_t *from, int idx) {
    int thread_idx = threadIdx.x;
    scalar_t *to = reinterpret_cast<scalar_t *>(data[0]) + block_work_size * idx;
    #pragma unroll
    for (int i = 0; i < thread_work_size; i++) {
      if (thread_idx >= remaining) {
        return;
      }
      int linear_idx = thread_idx + block_work_size * idx;
      int offset = output_offset_calculator.get(linear_idx)[0];
      storer.store(from[i], data[0], offset);
      thread_idx += num_threads;
    }
  }
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
      memory::detail::static_unroll<detail::multi_outputs_store_helper, num_outputs>::with_args(this->data, offsets, from[i]);
      thread_idx += num_threads;
    }
  }
};

}  // namespace policies

// This is only used in host, but we will wrap this into some templates
// which is C10_HOST_DEVICE, so we have to make this C10_HOST_DEVICE
// in order to compile
template<typename scalar_t>
inline C10_HOST_DEVICE int can_vectorize_up_to(char *pointer) {
  uint64_t address = reinterpret_cast<uint64_t>(pointer);
  constexpr int vec2_alignment = std::alignment_of<aligned_vector<scalar_t, 2>>::value;
  constexpr int vec4_alignment = std::alignment_of<aligned_vector<scalar_t, 4>>::value;
  if (address % vec4_alignment == 0) {
    return 4;
  } else if (address % vec2_alignment == 0) {
    return 2;
  }
  return 1;
}

template<int i>
struct can_vectorize_up_to_helper {
  template <typename array_t, typename traits>
  static C10_HOST_DEVICE void apply(int &result, array_t pointers, traits _) {
    using arg_t = typename traits::template arg<i>::type;
    // `pointers` hold the data_ptr for tensors [output, input0, input1, ...], so we
    // need a +1 offset to get the input
    result = std::min<int>(result, can_vectorize_up_to<arg_t>(pointers[i + 1]));
  }
};

template<typename func_t, typename array_t>
inline int can_vectorize_up_to(array_t pointers) {
  using traits = function_traits<func_t>;
  using return_t = typename traits::result_type;
  constexpr int arity = traits::arity;
  int result = can_vectorize_up_to<return_t>(pointers[0]);
  // We need to get the type for each argument of `func_t`, this can only
  // be done at compile time.
  detail::static_unroll<can_vectorize_up_to_helper, arity>::with_args(result, pointers, traits());
  return result;
}

}}} // namespace at::native::memory
