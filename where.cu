#include <ATen/native/cuda/Loops.cuh>
#include <helper.cuh>

std::vector<int64_t> shape = {
  5, 5, 5
};
std::vector<std::vector<int64_t>> strides = {
  // warning: strides are in bytes!
  {16, 80, 400},
  {1, 5, 25},
  {16, 80, 400},
  {0, 0, 0},
};
std::vector<at::ScalarType> dtypes = {
  at::ScalarType::ComplexDouble,
  at::ScalarType::Bool,
  at::ScalarType::ComplexDouble,
  at::ScalarType::ComplexDouble,
};
std::vector<char *> data_ptrs = {
  nullptr, nullptr, nullptr, nullptr,
};
bool is_contiguous = false;
int64_t noutputs = 1;

using namespace at;
using namespace at::native;

int main() {
  data_ptrs[0] = (char *)zeros<c10::complex<double>>(30);
  data_ptrs[1] = (char *)zeros<bool>(30);
  data_ptrs[2] = (char *)arange<c10::complex<double>>(30);
  data_ptrs[3] = (char *)arange<c10::complex<double>>(1);
  print((bool *)data_ptrs[1], 30);
  print((c10::complex<double> *)data_ptrs[2], 30);
  print((c10::complex<double> *)data_ptrs[3], 30);
  cudaDeviceSynchronize();
  TensorIteratorBase iter;  // uses the hardcoded globals above
  gpu_kernel(iter, [] GPU_LAMBDA (bool cond, c10::complex<double> a, c10::complex<double> b) {
    return cond ? a : b;
  });
  cudaDeviceSynchronize();
  print((c10::complex<double> *)data_ptrs[0], 30);
}
