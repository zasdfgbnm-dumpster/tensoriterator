#include <ATen/native/cuda/Loops.cuh>
#include <helper.cuh>

#include <cuda_runtime.h>

std::vector<int64_t> shape = {
  2, 3, 5
};
std::vector<std::vector<int64_t>> strides = {
  // warning: strides are in bytes!
  {4, 8, 24},
  {4, 8, 24},
  {4, 8, 24},
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
  data_ptrs[0] = (char *)zeros<float>(30);
  data_ptrs[1] = (char *)arange<int>(30);
  data_ptrs[2] = (char *)arange<float>(30);
  print((int *)data_ptrs[1], 30);
  print((float *)data_ptrs[2], 30);
  cudaDeviceSynchronize();
  TensorIteratorBase iter;  // uses the hardcoded globals above

  int niter = 1000;
  cudaEvent_t start, end;
  cudaEventCreate(&start);
  cudaEventCreate(&end);

  cudaEventRecord(start);

  for (int i = 0; i < niter; i++) {
    gpu_kernel(iter, [] GPU_LAMBDA (float a, float b) {
      return a + b;
    });
  }

  cudaEventRecord(end);
  cudaEventSynchronize(end);

  float tsf = 0;
  cudaEventElapsedTime(&tsf, start, end);

  printf("add\n"
         "time cost %.3e ms\n", tsf / niter);

  print((float *)data_ptrs[0], 30);
}
