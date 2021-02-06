#include <iostream>

template <typename T>
void echo_arange(int64_t size) {
  int64_t size_ = size * sizeof(T);

  // set
  T *buf = new T[size];
  for (int64_t i = 0; i < size; i++) {
    buf[i] = T(i);
  }
  T *dev;
  cudaMalloc(&dev, size_);
  cudaDeviceSynchronize();
  cudaMemcpy(dev, buf, size_, cudaMemcpyDefault);
  cudaDeviceSynchronize();
  delete [] buf;

  // print
  buf = new T[size];
  cudaDeviceSynchronize();
  cudaMemcpy(buf, dev, size_, cudaMemcpyDefault);
  cudaDeviceSynchronize();
  for (int64_t i = 0; i < size; i++) {
    std::cout << buf[i] << ", ";
  }
  std::cout << std::endl;
  delete [] buf;
}

int main() {
  echo_arange<float>(30);
}
