#include <iostream>

constexpr int64_t N = 5;
__managed__ float data[N];


#define CHECK() do { auto code = cudaGetLastError(); if(code != cudaSuccess) throw std::runtime_error(cudaGetErrorString(code)); } while(0)

struct useless {};

template<typename type, typename whatever>
struct base {
  type object;
  __device__ base(type obj, whatever unused): object(obj) {}
};

template <typename type>
struct derived : base<type, useless> {
  __device__ derived(type obj):
    base<type, useless>(obj, useless()) {}
};

struct echo {
  echo(): n(3) {}

  __device__ int get(int i) const {
    // this function just returns i
    int x = 0;

    if (n == 0) {
      return x;
    }
    x = i;

    return x;
  }

  int n;
  int whatever[50];
};


__global__ void range_kernel(float *data, echo obj) {
#ifdef BUG
  auto container = derived<echo>(obj);
  int offsets = container.object.get(blockIdx.x);
#else
  int offsets = obj.get(blockIdx.x);
#endif
  *(data + offsets) = blockIdx.x;
}

int main() {
  auto oc = echo();
  range_kernel<<<N, 1>>>(data, oc);
  cudaDeviceSynchronize();
  CHECK();
  for (int64_t i = 0; i < N; i++) {
    std::cout << data[i] << ", ";
  }
  std::cout << std::endl;
}
