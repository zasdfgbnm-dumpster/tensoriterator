#pragma once

#include <array>
#include <cstdint>
#include <iostream>
#include <vector>
#include <Array.h>
#include <THCIntegerDivider.cuh>

constexpr int MAX_DIMS = 25;
using index_t = uint32_t;
using offset_t = at::detail::Array<uint32_t, std::max<int>(2, 1)>;

struct OffsetCalculator {
  OffsetCalculator(int dims, const int64_t* sizes, const int64_t* const* strides) : dims(dims) {
    for (int i = 0; i < MAX_DIMS; ++i) {
      if (i < dims) {
        sizes_[i] = IntDivider<index_t>(sizes[i]);
      } else {
        sizes_[i] = IntDivider<index_t>(1);
      }
      for (int arg = 0; arg < 2; arg++) {
        strides_[i][arg] =  i < dims ? strides[arg][i] : 0;
      }
    }
  }

  __host__ __device__ offset_t get(index_t linear_idx) const {
    offset_t offsets;
    #pragma unroll
    for (int arg = 0; arg < 2; arg++) {
      offsets[arg] = 0;
    }

    #pragma unroll
    for (int dim = 0; dim < MAX_DIMS; ++dim) {
      if (dim == dims) {
        break;
      }
      auto divmod = sizes_[dim].divmod(linear_idx);
      linear_idx = divmod.div;

      #pragma unroll
      for (int arg = 0; arg < 2; arg++) {
        offsets[arg] += divmod.mod * strides_[dim][arg];
      }

    }
    return offsets;
  }

  int dims;
  IntDivider<index_t> sizes_[MAX_DIMS];
  index_t strides_[MAX_DIMS][2];
};
