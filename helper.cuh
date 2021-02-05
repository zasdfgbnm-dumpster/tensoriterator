#pragma once

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