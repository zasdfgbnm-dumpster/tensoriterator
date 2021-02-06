#pragma once

#include <array>
#include <cstdint>
#include <iostream>
#include <vector>
#include <Array.h>

constexpr int MAX_DIMS = 25;
using index_t = uint32_t;
using offset_t = at::detail::Array<uint32_t, std::max<int>(2, 1)>;

struct OffsetCalculator {
  OffsetCalculator(int dims) : dims(dims) {}

  __host__ __device__ offset_t get(index_t linear_idx) const {
    offset_t offsets;
    offsets[0] = 0;
    offsets[1] = 0;

    #pragma unroll
    for (int dim = 0; dim < MAX_DIMS; ++dim) {
      if (dim == dims) {
        break;
      }
      offsets[0] = linear_idx;
      offsets[1] = linear_idx;
    }
    return offsets;
  }

  int dims;
  int whatever[MAX_DIMS * 2];
};
