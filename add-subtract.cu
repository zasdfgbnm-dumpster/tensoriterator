#include <ATen/native/cuda/Loops.cuh>
#include <helper.cuh>

std::vector<int64_t> shape = {
  2, 3, 5
};
std::vector<std::vector<int64_t>> strides = {
  // warning: strides are in bytes!
  {4, 8, 24},
  {4, 8, 24},
  {4, 8, 24},
  {4, 8, 24},
};
std::vector<at::ScalarType> dtypes = {
  at::ScalarType::Float,
  at::ScalarType::Float,
  at::ScalarType::Float,
  at::ScalarType::Float,
};
std::vector<char *> data_ptrs = {
  nullptr, nullptr, nullptr, nullptr
};
bool is_contiguous = true;
int64_t noutputs = 2;

using namespace at;
using namespace at::native;

int main() {
  data_ptrs[0] = (char *)zeros<float>(30);
  data_ptrs[1] = (char *)zeros<float>(30);
  data_ptrs[2] = (char *)arange<float>(30);
  data_ptrs[3] = (char *)arange<float>(30);
  print((float *)data_ptrs[2], 30);
  print((float *)data_ptrs[3], 30);
  cudaDeviceSynchronize();
  TensorIteratorBase iter;  // uses the hardcoded globals above
  gpu_kernel_multiple_outputs(iter, [] GPU_LAMBDA (float a, float b) {
    return thrust::tuple<float, float>(a + b, a - b);
  });
  print((float *)data_ptrs[0], 30);
  print((float *)data_ptrs[1], 30);
}
