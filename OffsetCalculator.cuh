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
  OffsetCalculator() : dims(3) {}

  __host__ __device__ int get(index_t linear_idx) const {
    int offset;
    offset = 0;

    #pragma unroll
    for (int dim = 0; dim < MAX_DIMS; ++dim) {
      if (dim == dims) {
        break;
      }
      offset = linear_idx;
    }
    return offset;
  }

  int dims;
  int whatever[MAX_DIMS * 2];
};
