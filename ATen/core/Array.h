#pragma once

// A fixed-size array type usable from both host and
// device code.

namespace at {
namespace detail {

template <typename T, int size> struct Array {
  T data[size];

  __host__ __device__ T operator[](int i) const { return data[i]; }
  __host__ __device__ T &operator[](int i) { return data[i]; }
#ifdef __HIP_PLATFORM_HCC__
  __host__ __device__ Array() = default;
  __host__ __device__ Array(const Array &) = default;
  __host__ __device__ Array &operator=(const Array &) = default;
#else
  Array() = default;
  Array(const Array &) = default;
  Array &operator=(const Array &) = default;
#endif

  // Fill the array with x.
  __host__ __device__ Array(T x) {
    for (int i = 0; i < size; i++) {
      data[i] = x;
    }
  }
};

} // namespace detail
} // namespace at
