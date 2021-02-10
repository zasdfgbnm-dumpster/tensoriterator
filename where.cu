constexpr int num_threads = 64;
constexpr int thread_work_size = 4;
constexpr int block_work_size = num_threads * thread_work_size;

#include <c10/util/complex.h>
#include <ATen/native/cuda/MemoryAccess.cuh>

using namespace at;
using namespace at::native;

struct alignas(16) A {
  double data[2];
};

__launch_bounds__(num_threads)
__global__ void unrolled_elementwise_kernel(A *result, A *data)
{
  auto policy = memory::policies::unroll<A *>(data);
  
  using return_t = A;
  using args_t = std::tuple<bool, A, A>;

  int idx = blockIdx.x;

  return_t results[thread_work_size];
  args_t args[thread_work_size];

  // load
  policy.load(args, idx);

  if (idx >= 0) {
    return;
  }

  #pragma unroll
  for (int i = 0; i < thread_work_size; i++) {
    results[i] = std::get<1>(args[i]);
    *result = results[i];
  }
}

int main() {
  unrolled_elementwise_kernel<<<1, num_threads, 0>>>(nullptr, nullptr);
  cudaDeviceSynchronize();
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
