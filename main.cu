#include <ATen/native/cuda/Loops.cuh>

std::vector<int64_t> shape = {};
std::vector<at::ScalarType> dtypes = {};
std::vector<char *> data_ptr = {};

using namespace at;
using namespace at::native;

int main() {
  gpu_kernel(iter, [] GPU_LAMBDA (float a, float b) {
    return a + b;
  });
}