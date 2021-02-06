#include <helper.cuh>
#include <c10/macros/Macros.h>
#include <iostream>
#include <ATen/cuda/detail/OffsetCalculator.cuh>
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
__global__ void unrolled_elementwise_kernel_for_multi_outputs(int N, func_t f, array_t data, out_calc_t oc) {
  using return_t = thrust::tuple<float, float>;
  using args_t = std::tuple<float, float>;

  return_t results;
  args_t args;

  // load
  std::get<0>(args) = *(data[2] + blockIdx.x);
  std::get<1>(args) = *(data[3] + blockIdx.x);

  // compute
  results = f(std::get<0>(args), std::get<1>(args));

  // store
#ifdef BUG
  auto policy = C<out_calc_t>(oc);
  auto offsets = policy.offsets(blockIdx.x);
#else
  offset_t offsets = oc.get(blockIdx.x);
#endif
  *(data[0] + offsets[0]) = thrust::get<0>(results);
  *(data[1] + offsets[1]) = thrust::get<1>(results);
}

template <typename func_t, typename array_t, typename out_calc_t>
static inline void launch_unrolled_kernel_for_multi_outputs(int64_t N, const func_t& f, array_t data, out_calc_t oc) {
  unrolled_elementwise_kernel_for_multi_outputs<func_t, array_t><<<N, 1, 0>>>(N, f, data, oc);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}


template <typename func_t>
void gpu_kernel_multiple_outputs(const func_t& f) {
  using output_t = thrust::tuple<float, float>;

  at::detail::Array<float*, 4> data;
  for (int i = 0; i < 4; i++) {
    data[i] = data_ptrs[i];
  }

  int64_t numel = N;

  auto output_calc = make_output_offset_calculator();
  launch_unrolled_kernel_for_multi_outputs(numel, f, data, output_calc);
}

void compute() {
  data_ptrs[0] = zeros<float>(N);
  data_ptrs[1] = zeros<float>(N);
  gpu_kernel_multiple_outputs([] C10_HOST_DEVICE (float a, float b) {
    return thrust::tuple<float, float>(a + b, a - b);
  });
  cudaDeviceSynchronize();
  print(data_ptrs[0], N);
  print(data_ptrs[1], N);
  std::cout << std::endl;
}

int main() {
  data_ptrs[2] = arange<float>(N);
  data_ptrs[3] = arange<float>(N);
  print(data_ptrs[2], N);
  print(data_ptrs[3], N);
  std::cout << std::endl;
  compute();
}
