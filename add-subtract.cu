#include <helper.cuh>
#include <c10/macros/Macros.h>
#include <iostream>
#include <ATen/native/TensorIterator.h>
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
std::vector<char *> data_ptrs = {
  nullptr, nullptr, nullptr, nullptr
};
int64_t N = 5;

using namespace at;

static OffsetCalculator make_output_offset_calculator(const TensorIteratorBase& iter) {
  std::array<const int64_t*, 2> strides;
  int64_t element_sizes[2];
  for (int i = 0; i < 2; i++) {
    strides[i] = iter.strides(i).data();
    element_sizes[i] = sizeof(float);
  }
  return OffsetCalculator(iter.ndim(), iter.shape().data(), strides.data(), element_sizes);
}


template<typename out_calc_t, typename A>
struct B {
  out_calc_t output_offset_calculator;

  __device__ B(out_calc_t oc, A unused): output_offset_calculator(oc) {}
};

struct A {};

template <typename out_calc_t>
struct C : B<out_calc_t, A> {
  __device__ C( out_calc_t oc):
    B<out_calc_t, A>(oc, A()) {}

  __device__ inline offset_t offsets(int linear_idx) {
    return this->output_offset_calculator.get(linear_idx);
  }
};

template <typename func_t, typename array_t, typename out_calc_t>
C10_LAUNCH_BOUNDS_1(1)
__global__ void unrolled_elementwise_kernel_for_multi_outputs(int N, func_t f, array_t data, out_calc_t oc) {
  int remaining = N - blockIdx.x;
  auto policy = C<out_calc_t>(oc);

  using return_t = thrust::tuple<float, float>;
  using args_t = std::tuple<float, float>;

  int linear_idx = threadIdx.x + blockIdx.x;

  if (threadIdx.x >= remaining) {
    return;
  }

  return_t results;
  args_t args;

  // load
  std::get<0>(args) = *(data[2] + linear_idx);
  std::get<1>(args) = *(data[3] + linear_idx);

  // compute
  results = f(std::get<0>(args), std::get<1>(args));

  // store
#ifdef BUG
  auto offsets = policy.offsets(linear_idx);
#else
  offset_t offsets = oc.get(linear_idx);
#endif
  *(data[0] + offsets[0]) = thrust::get<0>(results);
  *(data[1] + offsets[1]) = thrust::get<1>(results);
}

template <typename func_t, typename array_t, typename out_calc_t>
static inline void launch_unrolled_kernel_for_multi_outputs(int64_t N, const func_t& f, array_t data, out_calc_t oc) {
  TORCH_INTERNAL_ASSERT(N > 0 && N <= std::numeric_limits<int32_t>::max());
  unrolled_elementwise_kernel_for_multi_outputs<func_t, array_t><<<N, 1, 0>>>(N, f, data, oc);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}


template <typename func_t>
void gpu_kernel_multiple_outputs(TensorIteratorBase& iter, const func_t& f) {
  using output_t = thrust::tuple<float, float>;

  at::detail::Array<float*, 4> data;
  for (int i = 0; i < 4; i++) {
    data[i] = (float*)iter.data_ptr(i);
  }

  int64_t numel = iter.numel();

  auto output_calc = make_output_offset_calculator(iter);
  launch_unrolled_kernel_for_multi_outputs(numel, f, data, output_calc);
}

void compute() {
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
  compute();
}
