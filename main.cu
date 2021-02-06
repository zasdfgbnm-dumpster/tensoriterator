#include <iostream>

struct bug {
  int zero = 0;
  int large_unused[50];

  __device__ void are_you_ok() const {
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
  derived(obj).obj.are_you_ok();
  obj.are_you_ok();
}

int main() {
  kernel<<<1, 1>>>(bug());
  cudaDeviceSynchronize();
}
