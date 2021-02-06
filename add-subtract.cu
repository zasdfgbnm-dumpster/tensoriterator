#include <helper.cuh>
#include <iostream>
#include <OffsetCalculator.cuh>

int64_t N = 5;

using namespace at;

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
  int offsets = container.object.get(blockIdx.x);
#else
  int offsets = oc.get(blockIdx.x);
#endif
  *(data + offsets) = blockIdx.x;
}

int main() {
  float *data = zeros<float>(N);
  auto oc = OffsetCalculator();
  range_kernel<<<N, 1>>>(data, oc);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  cudaDeviceSynchronize();
  print(data, N);
}
