
#pragma once

#include <c10/macros/Macros.h>
#include <iostream>

#define NUM_THREADS (C10_WARP_SIZE * 2)
#define BLOCK_WORK_SIZE NUM_THREADS

constexpr int num_threads = NUM_THREADS;
constexpr int block_work_size = BLOCK_WORK_SIZE;

#include <ATen/native/TensorIterator.h>
#include <ATen/cuda/detail/OffsetCalculator.cuh>

#include <thrust/tuple.h>

using namespace at;

static OffsetCalculator make_output_offset_calculator(const TensorIteratorBase& iter) {
  std::array<const int64_t*, 2> strides;
  int64_t element_sizes[2];
  for (int i = 0; i < 2; i++) {
    strides[i] = iter.strides(i).data();
    element_sizes[i] = sizeof(float);
  }
  return OffsetCalculator(iter.ndim(), iter.shape().data(), strides.data(), element_sizes);
}


template<typename out_calc_t, typename loader_t>
struct unroll {
  out_calc_t output_offset_calculator;

  __device__ unroll(out_calc_t oc, loader_t l): output_offset_calculator(oc) {}
};

struct LoadWithoutCast {};

template <typename out_calc_t>
struct multi_outputs_unroll : unroll<out_calc_t, LoadWithoutCast> {
  __device__ multi_outputs_unroll( out_calc_t oc):
    unroll<out_calc_t, LoadWithoutCast>(oc, LoadWithoutCast()) {}

  __device__ inline offset_t offsets(int linear_idx) {
    return this->output_offset_calculator.get(linear_idx);
  }
};

template <typename func_t, typename array_t, typename out_calc_t>
C10_LAUNCH_BOUNDS_1(num_threads)
__global__ void unrolled_elementwise_kernel_for_multi_outputs(int N, func_t f, array_t data, out_calc_t oc) {
  int remaining = N - block_work_size * blockIdx.x;
  auto policy = multi_outputs_unroll<out_calc_t>(oc);

  using return_t = thrust::tuple<float, float>;
  using args_t = std::tuple<float, float>;

  int linear_idx = threadIdx.x + block_work_size * blockIdx.x;

  if (threadIdx.x >= remaining) {
    return;
  }

  return_t results;
  args_t args;

  // load
  std::get<0>(args) = *(reinterpret_cast<float *>(data[2]) + linear_idx);
  std::get<1>(args) = *(reinterpret_cast<float *>(data[3]) + linear_idx);

  // compute
  results = f(std::get<0>(args), std::get<1>(args));

  // store
#ifdef BUG
  auto offsets = policy.offsets(linear_idx);
#else
  offset_t offsets = oc.get(linear_idx);
#endif
  *(reinterpret_cast<float *>(data[0]) + offsets[0]) = thrust::get<0>(results);
  *(reinterpret_cast<float *>(data[1]) + offsets[1]) = thrust::get<1>(results);
}

template <typename func_t, typename array_t, typename out_calc_t>
static inline void launch_unrolled_kernel_for_multi_outputs(int64_t N, const func_t& f, array_t data, out_calc_t oc) {
  TORCH_INTERNAL_ASSERT(N > 0 && N <= std::numeric_limits<int32_t>::max());
  int64_t grid = (N + block_work_size - 1) / block_work_size;
  unrolled_elementwise_kernel_for_multi_outputs<func_t, array_t><<<grid, num_threads, 0>>>(N, f, data, oc);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}


template <typename func_t>
void gpu_kernel_multiple_outputs(TensorIteratorBase& iter, const func_t& f) {
  using output_t = thrust::tuple<float, float>;

  at::detail::Array<char*, 4> data;
  for (int i = 0; i < 4; i++) {
    data[i] = (char*)iter.data_ptr(i);
  }

  int64_t numel = iter.numel();

  if (iter.is_contiguous()) {
    auto output_calc = TrivialOffsetCalculator();
    launch_unrolled_kernel_for_multi_outputs(numel, f, data, output_calc);
  } else {
    auto output_calc = make_output_offset_calculator(iter);
    launch_unrolled_kernel_for_multi_outputs(numel, f, data, output_calc);
  }
}
