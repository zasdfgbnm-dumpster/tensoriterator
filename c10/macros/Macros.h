#pragma once

#include <sstream>
#include <string>

// Designates functions callable from the host (CPU) and the device (GPU)
#define C10_HOST_DEVICE __host__ __device__
#define C10_DEVICE __device__
#define C10_HOST __host__
// constants from (https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#features-and-technical-specifications)
// The maximum number of threads per multiprocessor is 1024 for Turing architecture (7.5),
// 1536 for Geforce Ampere (8.6),
// and 2048 for all other architectures. You'll get warnings if you exceed these constants.
// Hence, the following macros adjust the input values from the user to resolve potential warnings.
#if __CUDA_ARCH__ == 750
constexpr uint32_t CUDA_MAX_THREADS_PER_SM = 1024;
#elif __CUDA_ARCH__ == 860
constexpr uint32_t CUDA_MAX_THREADS_PER_SM = 1536;
#else
constexpr uint32_t CUDA_MAX_THREADS_PER_SM = 2048;
#endif
// CUDA_MAX_THREADS_PER_BLOCK is same for all architectures currently
constexpr uint32_t CUDA_MAX_THREADS_PER_BLOCK = 1024;
// CUDA_THREADS_PER_BLOCK_FALLBACK is the "canonical fallback" choice of block size.
// 256 is a good number for this fallback and should give good occupancy and
// versatility across all architectures.
constexpr uint32_t CUDA_THREADS_PER_BLOCK_FALLBACK = 256;
// NOTE: if you are thinking of constexpr-ify the inputs to launch bounds, it
//       turns out that although __launch_bounds__ can take constexpr, it
//       can't take a constexpr that has anything to do with templates.
//       Currently we use launch_bounds that depend on template arguments in
//       Loops.cuh, Reduce.cuh and LossCTC.cuh. Hence, C10_MAX_THREADS_PER_BLOCK and
//       C10_MIN_BLOCKS_PER_SM are kept as macros.
// Suppose you were planning to write __launch_bounds__(a, b), based on your performance tuning on a modern GPU.
// Instead, you should write __launch_bounds__(C10_MAX_THREADS_PER_BLOCK(a), C10_MIN_BLOCKS_PER_SM(a, b)),
// which will also properly respect limits on old architectures.
#define C10_MAX_THREADS_PER_BLOCK(val) (((val) <= CUDA_MAX_THREADS_PER_BLOCK) ? (val) : CUDA_THREADS_PER_BLOCK_FALLBACK)
#define C10_MIN_BLOCKS_PER_SM(threads_per_block, blocks_per_sm) ((((threads_per_block)*(blocks_per_sm) <= CUDA_MAX_THREADS_PER_SM) ? (blocks_per_sm) : ((CUDA_MAX_THREADS_PER_SM + (threads_per_block) - 1) / (threads_per_block))))
// C10_LAUNCH_BOUNDS is analogous to __launch_bounds__
#define C10_LAUNCH_BOUNDS_0 __launch_bounds__(256, 4) // default launch bounds that should give good occupancy and versatility across all architectures.
#define C10_LAUNCH_BOUNDS_1(max_threads_per_block) __launch_bounds__((C10_MAX_THREADS_PER_BLOCK((max_threads_per_block))))
#define C10_LAUNCH_BOUNDS_2(max_threads_per_block, min_blocks_per_sm) __launch_bounds__((C10_MAX_THREADS_PER_BLOCK((max_threads_per_block))), (C10_MIN_BLOCKS_PER_SM((max_threads_per_block), (min_blocks_per_sm))))

#define C10_WARP_SIZE 32

#define CUDA_KERNEL_ASSERT(cond)                                               \
  if (C10_UNLIKELY(!(cond))) {                                                 \
    (void)(_wassert(_CRT_WIDE(#cond), _CRT_WIDE(__FILE__),                     \
                    static_cast<unsigned>(__LINE__)),                          \
           0);                                                                 \
  }

#define AT_ERROR(...)
#define TORCH_CHECK(...)
#define AT_ASSERT(...)
#define TORCH_INTERNAL_ASSERT(...)
#define C10_CUDA_KERNEL_LAUNCH_CHECK() do { auto code = cudaGetLastError(); if(code != cudaSuccess) {std::string e = cudaGetErrorString(code); std::cerr << e << std::endl; throw std::runtime_error(e);} } while(0)
