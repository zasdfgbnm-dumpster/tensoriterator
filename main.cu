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

  // allocate host array
  T *buf1 = new T[size];
  T *buf2 = new T[size];

  // allocate device array
  T *dev;
  cudaMalloc(&dev, bytes);

  // fill buf1 with arange
  for (int64_t i = 0; i < size; i++) {
    buf1[i] = T(i);
  }
  print_array(buf1, size);

  // copy buf1 -> dev -> buf2
  cudaDeviceSynchronize();
  cudaMemcpy(dev, buf1, bytes, cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();
  cudaMemcpy(buf2, dev, bytes, cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  print_array(buf2, size);

  delete [] buf1;
  delete [] buf2;
}

int main() {
  echo_arange<float>(30);
}
