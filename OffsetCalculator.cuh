#pragma once

#include <array>
#include <cstdint>
#include <iostream>
#include <vector>
constexpr int MAX_DIMS = 25;

struct OffsetCalculator {
  OffsetCalculator() : dims(3) {}

  __host__ __device__ int get(int i) const {
    int offset;
    offset = 0;

    #pragma unroll
    for (int dim = 0; dim < MAX_DIMS; ++dim) {
      if (dim == dims) {
        break;
      }
      offset = i;
    }
    return offset;
  }

  int dims;
  int whatever[MAX_DIMS * 2];
};
