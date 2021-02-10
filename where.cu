constexpr int num_threads = 64;
constexpr int thread_work_size = 4;

#include <ATen/native/cuda/MemoryAccess.cuh>

using namespace at;
using namespace at::native;

struct alignas(16) A {
  double data[2];
};

__global__ void unrolled_elementwise_kernel(A *result, A *data)
{
  auto policy = memory::policies::unroll<A *>(data);
  
  using return_t = A;
  using args_t = std::tuple<bool, A, A>;

  int idx = blockIdx.x;

  return_t results[4];
  args_t args[4];

  // load
  policy.load(args, idx);

  if (idx >= 0) {
    return;
  }

  #pragma unroll
  for (int i = 0; i < 4; i++) {
    results[i] = std::get<1>(args[i]);
    *result = results[i];
  }
}

int main() {
  unrolled_elementwise_kernel<<<1, 1>>>(nullptr, nullptr);
  cudaDeviceSynchronize();
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
