#include <ATen/native/cuda/Loops.cuh>
#include <helper.cuh>

std::vector<int64_t> shape = {
  4
};
std::vector<std::vector<int64_t>> strides = {
  // warning: strides are in bytes!
  {0},
  {0},
  {1},
};
std::vector<at::ScalarType> dtypes = {
  at::ScalarType::Bool,
  at::ScalarType::Long,
  at::ScalarType::Long,
};
std::vector<char *> data_ptrs = {
  nullptr, nullptr, nullptr,
};
bool is_contiguous = false;
int64_t noutputs = 1;

using namespace at;
using namespace at::native;

namespace {
  enum class CompareOpType {LE, GE, LT, GT};
}

template<typename scalar_t>
struct CompareFunctor{
  CompareFunctor(const CompareOpType op): op_(op) {}
  const CompareOpType op_;
  __device__ __forceinline__ bool operator() (scalar_t a, scalar_t b) const {
    if (op_ == CompareOpType::GE) {
      return a >= b;
    } else if (op_ == CompareOpType::GT) {
      return a > b;
    } else if (op_ == CompareOpType::LE) {
      return a <= b;
    } else { //LT
      return a < b;
    }
  }
};

void ge_kernel_cuda(TensorIteratorBase& iter) {
  using scalar_t = int64_t;
  gpu_kernel_with_scalars(iter, CompareFunctor<scalar_t>(CompareOpType::GE));
}

void gt_kernel_cuda(TensorIteratorBase& iter) {
  using scalar_t = int64_t;
  gpu_kernel_with_scalars(iter, CompareFunctor<scalar_t>(CompareOpType::GT));
}

void le_kernel_cuda(TensorIteratorBase& iter) {
  using scalar_t = int64_t;
  gpu_kernel_with_scalars(iter, CompareFunctor<scalar_t>(CompareOpType::LE));
}

void lt_kernel_cuda(TensorIteratorBase& iter) {
  using scalar_t = int64_t;
  gpu_kernel_with_scalars(iter, CompareFunctor<scalar_t>(CompareOpType::LT));
}

int main() {
  data_ptrs[0] = (char *)zeros<float>(4);
  data_ptrs[1] = (char *)full<int64_t>(1, 2);
  data_ptrs[2] = (char *)zeros<int64_t>(1);
  print((int64_t *)data_ptrs[1], 1);
  print((int64_t *)data_ptrs[2], 1);
  cudaDeviceSynchronize();
  TensorIteratorBase iter;  // uses the hardcoded globals above
  ge_kernel_cuda(iter);
  print((bool *)data_ptrs[0], 4);
}