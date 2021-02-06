#pragma once

#include <array>
#include <cstdint>
#include <iostream>
#include <c10/macros/Macros.h>
#include <ATen/core/Array.h>
#include <ATen/native/TensorIterator.h>
#include <THC/THCIntegerDivider.cuh>

constexpr int MAX_DIMS = 25;
using index_t = uint32_t;
using offset_t = at::detail::Array<uint32_t, std::max<int>(2, 1)>;

struct OffsetCalculator {
  // The offset for each argument. Wrapper around fixed-size array.
  // On CUDA, zero sized array is not allowed, so when we are handling nullary
  // operators, we need to create a size 1 offset to avoid compiler failure.
  // This size 1 offset is just a placeholder, and we will not use it.

  // if element_sizes is nullptr, then the strides will be in bytes, otherwise
  // the strides will be in # of elements.
  OffsetCalculator(int dims, const int64_t* sizes, const int64_t* const* strides, const int64_t* element_sizes=nullptr) : dims(dims) {
    TORCH_CHECK(dims <= MAX_DIMS, "tensor has too many (>", MAX_DIMS, ") dims");
    for (int i = 0; i < MAX_DIMS; ++i) {
      if (i < dims) {
        sizes_[i] = IntDivider<index_t>(sizes[i]);
      } else {
        sizes_[i] = IntDivider<index_t>(1);
      }
      for (int arg = 0; arg < 2; arg++) {
        int64_t element_size = (element_sizes == nullptr ? 1LL : element_sizes[arg]);
        strides_[i][arg] =  i < dims ? strides[arg][i] / element_size : 0;
      }
    }
  }

  C10_HOST_DEVICE offset_t get(index_t linear_idx) const {
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

struct TrivialOffsetCalculator {
  using offset_type = at::detail::Array<index_t, 2>;

  C10_HOST_DEVICE offset_type get(index_t linear_idx) const {
    offset_type offsets;
    #pragma unroll
    for (int arg = 0; arg < 2; arg++) {
      offsets[arg] = linear_idx;
    }
    return offsets;
  }
};
