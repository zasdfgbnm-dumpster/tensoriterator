
#pragma once

#include <c10/macros/Macros.h>
#include <iostream>

#define NUM_THREADS (C10_WARP_SIZE * 2)
#define THREAD_WORK_SIZE 4
#define BLOCK_WORK_SIZE (THREAD_WORK_SIZE * num_threads)

constexpr int num_threads = NUM_THREADS;
constexpr int thread_work_size = THREAD_WORK_SIZE;
constexpr int block_work_size = BLOCK_WORK_SIZE;

#include <c10/util/C++17.h>
#include <ATen/detail/FunctionTraits.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/TensorIteratorDynamicCasting.h>
#include <ATen/cuda/detail/OffsetCalculator.cuh>
#include <ATen/native/cuda/MemoryAccess.cuh>

#include <thrust/tuple.h>

#ifndef __HIP_PLATFORM_HCC__
#include <ATen/native/cuda/CUDALoops.cuh>
#else
#include <ATen/native/cuda/ROCmLoops.cuh>
#endif

namespace at { namespace native {

void gpu_kernel(TensorIteratorBase& iter) {
  launch_unrolled_kernel();
}

}} //namespace at::native
