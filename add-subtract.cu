#include <iostream>

constexpr int64_t N = 5;
__managed__ float data[N];

#define CHECK() do { auto code = cudaGetLastError(); if(code != cudaSuccess) throw std::runtime_error(cudaGetErrorString(code)); } while(0)

struct echo {
  int three = 3;
  int large_unused[50];

  __device__ int get(int i) const {
    // this function just returns i
    if (three == 0) {
      return 0;
    }
    return i;
  }
};

struct useless {};

struct base {
  echo obj;
  __device__ base(echo obj, useless unused): obj(obj) {}
};

struct derived : base {
  __device__ derived(echo obj):
    base(obj, useless()) {}
};


__global__ void range_kernel(float *data, echo obj) {
#ifdef BUG
  int offsets = derived(obj).obj.get(blockIdx.x);
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
