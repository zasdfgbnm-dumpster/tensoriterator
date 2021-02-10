constexpr int num_threads = 64;
constexpr int thread_work_size = 4;
constexpr int block_work_size = num_threads * thread_work_size;

#include <c10/util/complex.h>
#include <ATen/native/cuda/MemoryAccess.cuh>

using namespace at;
using namespace at::native;

C10_LAUNCH_BOUNDS_1(num_threads)
__global__ void unrolled_elementwise_kernel(c10::complex<double> *result, c10::complex<double> *data)
{
  auto policy = memory::policies::unroll<c10::complex<double> *>(data);
  
  using return_t = c10::complex<double>;
  using args_t = std::tuple<bool, c10::complex<double>, c10::complex<double>>;

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
