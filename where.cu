#include <iostream>
#include <tuple>

struct alignas(16) A {
  double data[2];
};

template<int arg_index>
struct unroll_load_helper {
  template <typename args_t>
  static __device__ void apply(args_t *args, int j) {
    printf("%d%d\n", 0, 0);
    std::get<arg_index>(args[j]) = {};
  }
};

struct static_unroll {
  template<typename... Args>
  static inline __host__ __device__ void with_args(Args&&... args) {
    unroll_load_helper<0>::apply(std::forward<Args>(args)...);
    unroll_load_helper<1>::apply(std::forward<Args>(args)...);
  }
};

__global__ void unrolled_elementwise_kernel(A *result)
{
  std::tuple<bool, A> args[2];
  #pragma unroll
  for (int i = 0; i < 2; i++) {
    static_unroll::with_args(args, i);
  }

  if ((int)blockIdx.x >= 0) {
    return;
  }
  // code below will not be executed
  *result = std::get<1>(args[0]);
}

int main() {
  unrolled_elementwise_kernel<<<1, 1>>>(nullptr);
  cudaDeviceSynchronize();
  auto code = cudaGetLastError();
  if(code != cudaSuccess) {
    std::string e = cudaGetErrorString(code);
    std::cerr << e << std::endl;
    throw std::runtime_error(e);
  }
}
