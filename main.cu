#include <vector>
#include <iostream>

std::vector<char *> data_ptrs = {
  nullptr, nullptr
};

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
void print(T *data, int64_t size) {
  T *buf = new T[size];
  int64_t size_ = size * sizeof(T);
  cudaDeviceSynchronize();
  cudaMemcpy(buf, data, size_, cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  for (int64_t i = 0; i < size; i++) {
    std::cout << buf[i] << ", ";
  }
  std::cout << std::endl;
  delete [] buf;
}

int main() {
  data_ptrs[0] = (char *)arange<float>(30);
  data_ptrs[1] = (char *)arange<float>(30);
  cudaDeviceSynchronize();
  print((float *)data_ptrs[0], 30);
  print((float *)data_ptrs[1], 30);
}
