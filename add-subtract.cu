#include <helper.cuh>
#include <iostream>
#include <OffsetCalculator.cuh>
#include <thrust/tuple.h>

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
std::vector<float *> data_ptrs = {
  nullptr, nullptr, nullptr, nullptr
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

template <typename func_t, typename array_t, typename out_calc_t>
__global__ void unrolled_elementwise_kernel_for_multi_outputs(array_t data, out_calc_t oc) {
  thrust::tuple<float, float> results;
#ifdef BUG
  auto policy = C<out_calc_t>(oc);
  auto offsets = policy.offsets(blockIdx.x);
#else
  offset_t offsets = oc.get(blockIdx.x);
#endif
  *(data[0] + offsets[0]) = blockIdx.x;
  *(data[1] + offsets[1]) = blockIdx.x;
}

template <typename func_t>
void gpu_kernel_multiple_outputs(const func_t& f) {
  at::detail::Array<float*, 4> data;
  for (int i = 0; i < 4; i++) {
    data[i] = data_ptrs[i];
  }

  auto oc = make_output_offset_calculator();
  unrolled_elementwise_kernel_for_multi_outputs<<<N, 1, 0>>>(data, oc);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

int main() {
  data_ptrs[0] = zeros<float>(N);
  data_ptrs[1] = zeros<float>(N);
  gpu_kernel_multiple_outputs([] __host__ __device__ (float a, float b) {
    return thrust::tuple<float, float>(a + b, a - b);
  });
  cudaDeviceSynchronize();
  print(data_ptrs[0], N);
  print(data_ptrs[1], N);
}
