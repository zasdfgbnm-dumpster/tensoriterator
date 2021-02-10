constexpr int num_threads = 64;
constexpr int thread_work_size = 4;

#include <iostream>
#include <c10/macros/Macros.h>

template<template<int i> typename func, int end, int current=0>
struct static_unroll {
  template<typename... Args>
  static inline __host__ __device__ void with_args(Args&&... args) {
    func<current>::apply(std::forward<Args>(args)...);
    static_unroll<func, end, current+1>::with_args(args...);
  }
};

template<template<int i> typename func, int end>
struct static_unroll<func, end, end> {
  template<typename... Args>
  static inline __host__ __device__ void with_args(Args... args) {}
};

// helper structs to be used with static_unroll to load arguments
// one by one

template<int arg_index>
struct unroll_load_helper {
  template <typename args_t, typename policy_t>
  static __device__ void apply(policy_t &self, args_t *args, int j) {
    auto addr = reinterpret_cast<uint64_t>(self.data);
    printf("address: %llu, mod: %llu\n", addr, addr % 16);
    std::get<arg_index>(args[j]) = {};
  }
};

// Assumption:
// all tensors are contiguous, that is: stride == sizeof(type) for all tensors
template<typename data_t, int num_outputs = 1>
struct unroll {
  data_t data;

  __device__ unroll(data_t data):
    data(data) {}

  template<typename args_t>
  __device__ inline void load(args_t *args, int idx) {
    constexpr int arity = std::tuple_size<args_t>::value;
    #pragma unroll
    for (int i = 0; i < thread_work_size; i++) {
      static_unroll<unroll_load_helper, arity>::with_args(*this, args, i);
    }
  }
};

struct alignas(16) A {
  double data[2];
};

__global__ void unrolled_elementwise_kernel(A *result, A *data)
{
  auto policy = unroll<A *>(data);
  
  using return_t = A;
  using args_t = std::tuple<bool, A, A>;

  int idx = blockIdx.x;

  return_t results[4];
  args_t args[4];

  // load
  policy.load(args, idx);

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
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
