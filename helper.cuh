#pragma once

#include <iostream>
#include <c10/macros/Macros.h>

template <typename T>
T *arange(int64_t size) {
  T *buf = new T[size];
  for (int64_t i = 0; i < size; i++) {
    buf[i] = T(i);
  }
  T *ret;
  int64_t size_ = size * sizeof(T);
  cudaMalloc(&ret, size_);
  cudaMemcpy(ret, buf, size_, cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();
  delete [] buf;
  // who cares about cudaFree :P LOL
  return ret;
}

template <typename T>
T *zeros(int64_t size) {
  T *buf = new T[size];
  for (int64_t i = 0; i < size; i++) {
    buf[i] = 0;
  }
  T *ret;
  int64_t size_ = size * sizeof(T);
  cudaMalloc(&ret, size_);
  cudaMemcpy(ret, buf, size_, cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();
  delete [] buf;
  // who cares about cudaFree :P LOL
  return ret;
}

template <typename T>
void print(T *data, int64_t size) {
  T *buf = new T[size];
  int64_t size_ = size * sizeof(T);
  cudaDeviceSynchronize();
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  cudaMemcpy(buf, data, size_, cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  for (int64_t i = 0; i < size; i++) {
    std::cout << buf[i] << ", ";
  }
  std::cout << std::endl;
  delete [] buf;
}