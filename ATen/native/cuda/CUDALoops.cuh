#pragma once

// This file provides two functions to help write GPU elementwise kernels:
//
//   gpu_kernel(TensorIterator iter, <lambda>)
//   gpu_kernel_with_scalars(TensorIterator iter, <lambda>)
//
// The gpu_kernel_with_scalars generates specializations that support a
// single scalar CPU argument, such as from `cuda_tensor + 5`. The CPU scalar
// is lifted to a kernel parameter instead of copying to device memory.
// This should be  used in conjunction with TensorIterator::allow_cpu_scalars_,
// which is the default for TensorIterator::binary_op. Otherwise, all inputs
// and the output must be on the GPU.
//
// For example, to write a reciprocal kernel for GPU float Tensors:
//
//   gpu_kernel(iter, []GPU_LAMBDA(float a) {
//    return 1.0f / a;
//   });
//
// To write a multiplication kernel for GPU float Tensors where one argument
// may be a CPU scalar:
//
//   gpu_kernel_with_scalars(iter, []GPU_LAMBDA(float a, float b) {
//     return a * b;
//   });
//
// See BinaryOpsKernel.cu for the complete implementation
//

#include <type_traits>
#include <tuple>
#include <iostream>
#include <c10/macros/Macros.h>
#include <c10/core/ScalarType.h>
#include <c10/util/TypeCast.h>
#include <ATen/core/Array.h>
#include <ATen/detail/FunctionTraits.h>

// Marks a lambda as executable on both the host and device. The __host__
// attribute is important so that we can access static type information from
// the host, even if the function is typically only executed on the device.
#ifndef GPU_LAMBDA
#define GPU_LAMBDA __host__ __device__
#endif

#ifdef __NVCC__
#define ASSERT_HOST_DEVICE_LAMBDA(type) \
  static_assert(__nv_is_extended_host_device_lambda_closure_type(type), \
                #type " must be a __host__ __device__ lambda")
#else
#define ASSERT_HOST_DEVICE_LAMBDA(type)
#endif


namespace at { namespace native {

C10_LAUNCH_BOUNDS_1(num_threads)
__global__ void unrolled_elementwise_kernel(c10::complex<double> *result, c10::complex<double> *data)
{
  auto policy = memory::policies::unroll<c10::complex<double> *>(data);
  
  using return_t = c10::complex<double>;
  using args_t = std::tuple<bool, c10::complex<double>, c10::complex<double>>;

  int idx = blockIdx.x;

  return_t results[thread_work_size];
  args_t args[thread_work_size];

  // load
  policy.load(args, idx);

  if (idx >= 0) {
    return;
  }

  #pragma unroll
  for (int i = 0; i < thread_work_size; i++) {
    results[i] = std::get<1>(args[i]);
    *result = results[i];
  }
}

static inline void launch_unrolled_kernel()
{
  TORCH_INTERNAL_ASSERT(N > 0 && N <= std::numeric_limits<int32_t>::max());
  int64_t grid = 1;
  unrolled_elementwise_kernel<<<grid, num_threads, 0>>>(nullptr, nullptr);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}} // namespace at::native
