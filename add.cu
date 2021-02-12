#include <ATen/native/cuda/Loops.cuh>
#include <helper.cuh>

#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

std::vector<int64_t> shape = {
  200, 300, 10000
};
std::vector<std::vector<int64_t>> strides = {
  // warning: strides are in bytes!
  {4, 800, 240000},
  {4, 800, 240000},
  {4, 800, 240000},
};
std::vector<at::ScalarType> dtypes = {
  at::ScalarType::Float,
  at::ScalarType::Int,
  at::ScalarType::Float,
};
std::vector<char *> data_ptrs = {
  nullptr, nullptr, nullptr,
};
bool is_contiguous = true;
int64_t noutputs = 1;

using namespace at;
using namespace at::native;

int main() {
  data_ptrs[0] = (char *)zeros<float>(600000000);
  data_ptrs[1] = (char *)arange<int>(600000000);
  data_ptrs[2] = (char *)arange<float>(600000000);
  print((int *)data_ptrs[1], 30);
  print((float *)data_ptrs[2], 30);
  cudaDeviceSynchronize();
  TensorIteratorBase iter;  // uses the hardcoded globals above

  int niter = 1000;

  cudaProfilerStart();
  std::cout << iter.numel() << std::endl;

  for (int i = 0; i < niter; i++) {
    gpu_kernel(iter, [] GPU_LAMBDA (float a, float b) {
      return a + b;
    });
  }

  cudaProfilerStop();

  print((float *)data_ptrs[0], 30);
}
