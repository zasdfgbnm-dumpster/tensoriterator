#include <ATen/native/cuda/Loops.cuh>

std::vector<int64_t> shape = {};
std::vector<std::vector<int64_t>> strides = {};
std::vector<at::ScalarType> dtypes = {};
std::vector<char *> data_ptrs = {};
bool is_contiguous = true;
int64_t noutputs = 1;

using namespace at;
using namespace at::native;

int main() {
  TensorIteratorBase iter;
  gpu_kernel(iter, [] GPU_LAMBDA (float a, float b) {
    return a + b;
  });
}