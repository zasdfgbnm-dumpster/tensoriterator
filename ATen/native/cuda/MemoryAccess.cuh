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
  template <typename args_t, typename policy_t>
  static __device__ void apply(policy_t &self, args_t *args, int j, int num_outputs) {
    using arg_t = std::tuple_element_t<arg_index, args_t>;
    // `data` hold the data_ptr for tensors [output, input0, input1, ...], so we
    // need a +1 offset to get the input
    auto addr = reinterpret_cast<uint64_t>(self.data);
    printf("address: %llu, mod: %llu\n", addr, addr % 16);
    std::get<arg_index>(args[j]) = 0;
  }
};

}  // namespace detail

namespace policies {

// Assumption:
// all tensors are contiguous, that is: stride == sizeof(type) for all tensors
template<typename data_t, int num_outputs = 1>
struct unroll {
  data_t data;
  int remaining;

  __device__ unroll(data_t data, int remaining):
    data(data), remaining(remaining) {}

  template<typename args_t>
  __device__ inline void load(args_t *args, int idx) {
    constexpr int arity = std::tuple_size<args_t>::value;
    int thread_idx = threadIdx.x;
    #pragma unroll
    for (int i = 0; i < thread_work_size; i++) {
      if (thread_idx >= remaining) {
        return;
      }
      detail::static_unroll<detail::unroll_load_helper, arity>::with_args(*this, args, i, num_outputs);
      thread_idx += num_threads;
    }
  }
};

}  // namespace policies

}}} // namespace at::native::memory
