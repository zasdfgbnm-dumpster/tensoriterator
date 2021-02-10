#include <iostream>
#include <tuple>

constexpr int thread_work_size = 4;

struct alignas(16) A {
  double data[2];
};

template<template<int i> typename func>
struct static_unroll {
  template<typename... Args>
  static inline __host__ __device__ void with_args(Args&&... args) {
    func<0>::apply(std::forward<Args>(args)...);
    func<1>::apply(std::forward<Args>(args)...);
    func<2>::apply(std::forward<Args>(args)...);
  }
};

// helper structs to be used with static_unroll to load arguments
// one by one

template<int arg_index>
struct unroll_load_helper {
  template <typename args_t>
  static __device__ void apply(args_t *args, int j) {
    uint64_t addr = 0;
    printf("address: %llu, mod: %llu\n", addr, addr % 16);
    std::get<arg_index>(args[j]) = {};
  }
};

__global__ void unrolled_elementwise_kernel(A *result, A *data)
{
  int idx = blockIdx.x;

  A results[4];
  std::tuple<bool, A, A> args[4];

  // load
  #pragma unroll
  for (int i = 0; i < thread_work_size; i++) {
    static_unroll<unroll_load_helper>::with_args(args, i);
  }

  if (idx >= 0) {
    return;
  }

  #pragma unroll
  for (int i = 0; i < 4; i++) {
    results[i] = std::get<1>(args[i]);
    *result = results[i];
  }
}

int main() {
  unrolled_elementwise_kernel<<<1, 1>>>(nullptr, nullptr);
  cudaDeviceSynchronize();
  auto code = cudaGetLastError();
  if(code != cudaSuccess) {
    std::string e = cudaGetErrorString(code);
    std::cerr << e << std::endl;
    throw std::runtime_error(e);
  }
}
