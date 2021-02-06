#pragma once

#include <iostream>

#define C10_CUDA_KERNEL_LAUNCH_CHECK() do { auto code = cudaGetLastError(); if(code != cudaSuccess) throw std::runtime_error(cudaGetErrorString(code)); } while(0)

template <typename T>
T *arange(int64_t size) {
  T *buf = new T[size];
  for (int64_t i = 0; i < size; i++) {
    buf[i] = T(i + 1);
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