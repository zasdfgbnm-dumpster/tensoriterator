
#pragma once

#include <c10/macros/Macros.h>
#include <iostream>

#define NUM_THREADS (C10_WARP_SIZE * 2)
#define BLOCK_WORK_SIZE NUM_THREADS

constexpr int num_threads = NUM_THREADS;
constexpr int block_work_size = BLOCK_WORK_SIZE;

#include <ATen/native/TensorIterator.h>
#include <ATen/cuda/detail/OffsetCalculator.cuh>
#include <ATen/native/cuda/MemoryAccess.cuh>

#include <thrust/tuple.h>

using namespace at;
using namespace at::native;

static OffsetCalculator make_input_offset_calculator(const TensorIteratorBase& iter) {
  std::array<const int64_t*, 2> strides;
  int64_t element_sizes[2];
  for (int i = 0; i < 2; i++) {
    strides[i] = iter.strides(i + iter.noutputs()).data();
    element_sizes[i] = sizeof(float);
  }
  return OffsetCalculator(iter.ndim(), iter.shape().data(), strides.data(), element_sizes);
}

static OffsetCalculator make_output_offset_calculator(const TensorIteratorBase& iter) {
  std::array<const int64_t*, 2> strides;
  int64_t element_sizes[2];
  for (int i = 0; i < 2; i++) {
    strides[i] = iter.strides(i).data();
    element_sizes[i] = sizeof(float);
  }
  return OffsetCalculator(iter.ndim(), iter.shape().data(), strides.data(), element_sizes);
}

template <typename func_t, typename array_t, typename out_calc_t>
C10_LAUNCH_BOUNDS_1(num_threads)
__global__ void unrolled_elementwise_kernel_for_multi_outputs(int N, func_t f, array_t data, out_calc_t oc) {
  int remaining = N - block_work_size * blockIdx.x;
  auto policy = memory::policies::multi_outputs_unroll<array_t, out_calc_t>(data, oc);

  using return_t = thrust::tuple<float, float>;
  using args_t = std::tuple<float, float>;

  int idx = blockIdx.x;

  if (threadIdx.x >= remaining) {
    return;
  }

  return_t results;
  args_t args;

  // load
  int linear_idx = threadIdx.x + block_work_size * idx;
  std::get<0>(args) = *(reinterpret_cast<float *>(data[2]) + linear_idx);
  std::get<1>(args) = *(reinterpret_cast<float *>(data[3]) + linear_idx);

  // compute
  results = f(std::get<0>(args), std::get<1>(args));

  // store
  policy.store(results, idx);
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
