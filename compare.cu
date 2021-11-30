#include <ATen/native/cuda/Loops.cuh>
#include <helper.cuh>

std::vector<int64_t> shape = {
  4
};
std::vector<std::vector<int64_t>> strides = {
  // warning: strides are in bytes!
  {1},
  {0},
};
std::vector<at::ScalarType> dtypes = {
  at::ScalarType::Bool,
  at::ScalarType::Long,
};
std::vector<char *> data_ptrs = {
  nullptr, nullptr,
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


int main() {
  data_ptrs[0] = (char *)zeros<bool>(4);
  data_ptrs[1] = (char *)full<int64_t>(1, 2);
  print((int64_t *)data_ptrs[1], 1);
  cudaDeviceSynchronize();
  auto f = CompareFunctor<int64_t>(CompareOpType::GE);
  BUnaryFunctor<decltype(f)> bf(f, 0L);
  TensorIteratorBase iter;  // uses the hardcoded globals above
  gpu_kernel(iter, bf);
  print((bool *)data_ptrs[0], 4);
}