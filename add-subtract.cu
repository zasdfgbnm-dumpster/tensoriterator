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
std::vector<char *> data_ptrs = {
  nullptr, nullptr, nullptr, nullptr
};
bool is_contiguous = true;
int64_t noutputs = 2;
int64_t N = 5;

using namespace at;
using namespace at::native;

void compute() {
  std::cout << "is_contiguous = " << is_contiguous << std::endl;
  data_ptrs[0] = (char *)zeros<float>(N);
  data_ptrs[1] = (char *)zeros<float>(N);
  TensorIteratorBase iter;  // uses the hardcoded globals above
  gpu_kernel_multiple_outputs(iter, [] C10_HOST_DEVICE (float a, float b) {
    return thrust::tuple<float, float>(a + b, a - b);
  });
  cudaDeviceSynchronize();
  print((float *)data_ptrs[0], N);
  print((float *)data_ptrs[1], N);
  std::cout << std::endl;
}

int main() {
  data_ptrs[2] = (char *)arange<float>(N);
  data_ptrs[3] = (char *)arange<float>(N);
  print((float *)data_ptrs[2], N);
  print((float *)data_ptrs[3], N);
  std::cout << std::endl;

  is_contiguous = true;
  compute();

  is_contiguous = false;
  compute();
}
