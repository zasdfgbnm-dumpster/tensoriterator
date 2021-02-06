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

struct useless {};

template<typename type, typename whatever>
struct container_base {
  type object;
  __device__ container_base(type obj, whatever unused): object(obj) {}
};

template <typename type>
struct container_derived : container_base<type, useless> {
  __device__ container_derived(type obj):
    container_base<type, useless>(obj, useless()) {}
};

template <typename out_calc_t>
__global__ void range_kernel(float *data, out_calc_t oc) {
#ifdef BUG
  auto container = container_derived<out_calc_t>(oc);
  offset_t offsets = container.object.get(blockIdx.x);
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
