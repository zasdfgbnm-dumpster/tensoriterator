#include <ATen/native/cuda/Loops.cuh>

using namespace at;
using namespace at::native;

int main() {
  unrolled_elementwise_kernel<<<1, num_threads, 0>>>(nullptr, nullptr);
  cudaDeviceSynchronize();
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
