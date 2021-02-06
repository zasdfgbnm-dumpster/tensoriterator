#include <iostream>

constexpr int64_t N = 5;
__managed__ float data[N];

#define CHECK() do { auto code = cudaGetLastError(); if(code != cudaSuccess) throw std::runtime_error(cudaGetErrorString(code)); } while(0)

struct echo {
  int zero = 0;
  int large_unused[50];

  __device__ void get() const {
    // this function just returns i
    if (zero == 0) {
      return;
    }
    printf("I have a bug\n");
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


__global__ void range_kernel(echo obj) {
#ifdef BUG
  derived(obj).obj.get();
#else
  obj.get();
#endif
}

int main() {
  range_kernel<<<1, 1>>>(echo());
  cudaDeviceSynchronize();
  CHECK();
}
