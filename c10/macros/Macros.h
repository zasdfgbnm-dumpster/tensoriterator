#pragma once

#include <sstream>
#include <string>

// Designates functions callable from the host (CPU) and the device (GPU)
#define C10_HOST_DEVICE __host__ __device__
#define C10_DEVICE __device__
#define C10_HOST __host__

// C10_LAUNCH_BOUNDS is analogous to __launch_bounds__
#define C10_LAUNCH_BOUNDS_0                                                    \
  __launch_bounds__(256,                                                       \
                    4) // default launch bounds that should give good occupancy
                       // and versatility across all architectures.
#define C10_LAUNCH_BOUNDS_1(max_threads_per_block)                             \
  __launch_bounds__((C10_MAX_THREADS_PER_BLOCK((max_threads_per_block))))
#define C10_LAUNCH_BOUNDS_2(max_threads_per_block, min_blocks_per_sm)          \
  __launch_bounds__(                                                           \
      (C10_MAX_THREADS_PER_BLOCK((max_threads_per_block))),                    \
      (C10_MIN_BLOCKS_PER_SM((max_threads_per_block), (min_blocks_per_sm))))

#define C10_WARP_SIZE 32

#define CUDA_KERNEL_ASSERT(cond)                                               \
  if (C10_UNLIKELY(!(cond))) {                                                 \
    (void)(_wassert(_CRT_WIDE(#cond), _CRT_WIDE(__FILE__),                     \
                    static_cast<unsigned>(__LINE__)),                          \
           0);                                                                 \
  }
