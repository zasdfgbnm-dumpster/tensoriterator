#include <iostream>

constexpr int64_t N = 5;
__managed__ float data[N];

#define CHECK() do { auto code = cudaGetLastError(); if(code != cudaSuccess) throw std::runtime_error(cudaGetErrorString(code)); } while(0)

struct bug {
  int zero = 0;
  int large_unused[50];

  __device__ void are_you_ok() const {
    // this function just returns i
    if (zero == 0) {
      printf("I am fine, thank you!\n");
      return;
    }
    printf("No, I have a bug!\n");
  }
};

struct useless {};

struct base {
  bug obj;
  __device__ base(bug obj, useless unused): obj(obj) {}
};

struct derived : base {
  __device__ derived(bug obj):
    base(obj, useless()) {}
};


__global__ void kernel(bug obj) {
#ifdef BUG
  derived(obj).obj.are_you_ok();
#else
  obj.are_you_ok();
#endif
}

int main() {
  kernel<<<1, 1>>>(bug());
  cudaDeviceSynchronize();
  CHECK();
}
