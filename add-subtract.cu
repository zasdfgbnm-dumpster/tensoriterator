#include <helper.cuh>
#include <iostream>
#include <OffsetCalculator.cuh>

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
int64_t N = 5;

using namespace at;

static OffsetCalculator make_output_offset_calculator() {
  std::array<const int64_t*, 2> strides;
  int64_t element_sizes[2];
  for (int i = 0; i < 2; i++) {
    strides[i] = ::strides[i].data();
    element_sizes[i] = sizeof(float);
  }
  return OffsetCalculator(shape.size(), shape.data(), strides.data(), element_sizes);
}

struct Useless {};

template<typename out_calc_t, typename A>
struct B {
  out_calc_t output_offset_calculator;

  __device__ B(out_calc_t oc, A unused): output_offset_calculator(oc) {}
};

template <typename out_calc_t>
struct C : B<out_calc_t, Useless> {
  __device__ C(out_calc_t oc):
    B<out_calc_t, Useless>(oc, Useless()) {}

  __device__ inline offset_t offsets(int linear_idx) {
    return this->output_offset_calculator.get(linear_idx);
  }
};

template <typename out_calc_t>
__global__ void range_kernel(float *data, out_calc_t oc) {
#ifdef BUG
  auto policy = C<out_calc_t>(oc);
  auto offsets = policy.offsets(blockIdx.x);
#else
  offset_t offsets = oc.get(blockIdx.x);
#endif
  *(data + offsets[0]) = blockIdx.x;
}

int main() {
  float *data = zeros<float>(N);
  auto oc = make_output_offset_calculator();
  range_kernel<<<N, 1>>>(data, oc);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  cudaDeviceSynchronize();
  print(data, N);
}
