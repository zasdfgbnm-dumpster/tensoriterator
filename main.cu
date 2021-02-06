#include <iostream>

template <typename T>
void print_array(T *buf, int64_t size) {
  for (int64_t i = 0; i < size; i++) {
    std::cout << buf[i] << ", ";
  }
  std::cout << std::endl;
}

template <typename T>
void echo_arange(int64_t size) {
  int64_t bytes = size * sizeof(T);

  // set
  T *buf = new T[size];
  for (int64_t i = 0; i < size; i++) {
    buf[i] = T(i);
  }
  print_array(buf, size);
  T *dev;
  cudaMalloc(&dev, bytes);
  cudaDeviceSynchronize();
  cudaMemcpy(dev, buf, bytes, cudaMemcpyDefault);
  cudaDeviceSynchronize();
  delete [] buf;

  // print
  buf = new T[size];
  cudaDeviceSynchronize();
  cudaMemcpy(buf, dev, bytes, cudaMemcpyDefault);
  cudaDeviceSynchronize();
  print_array(buf, size);
  delete [] buf;
}

int main() {
  echo_arange<float>(30);
}
