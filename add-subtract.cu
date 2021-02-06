#include <helper.cuh>
#include <iostream>

int64_t N = 5;
constexpr int MAX_DIMS = 25;

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

struct OffsetCalculator {
  OffsetCalculator() : dims(3) {}

  __host__ __device__ int get(int i) const {
    int offset;
    offset = 0;

    #pragma unroll
    for (int dim = 0; dim < MAX_DIMS; ++dim) {
      if (dim == dims) {
        break;
      }
      offset = i;
    }
    return offset;
  }

  int dims;
  int whatever[MAX_DIMS * 2];
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
