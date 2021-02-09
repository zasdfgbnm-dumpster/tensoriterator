#pragma once

#include <c10/macros/Macros.h>
#include <c10/util/complex.h>
#include <c10/core/ScalarType.h>

#define ERROR_UNSUPPORTED_CAST assert(false);

namespace c10 {

template<typename dest_t, typename src_t>
struct needs_real {
  constexpr static bool value = (is_complex<src_t>::value && !is_complex<dest_t>::value);
};

template<bool, typename src_t>
struct maybe_real {
  C10_HOST_DEVICE static inline src_t apply(src_t src) {
    return src;
  }
};

template<typename src_t>
struct maybe_real<true, src_t> {
  C10_HOST_DEVICE static inline decltype(auto) apply(src_t src) {
    return src.real();
  }
};

// Note: deliberately ignores undefined behavior, consistent with NumPy.
// PyTorch's type conversions can cause a variety of undefined behavior,
// including float to integral overflow and signed to unsigned integer overflow.
// Some of this undefined behavior is addressed below.
template <typename dest_t, typename src_t>
struct static_cast_with_inter_type {
  C10_HOST_DEVICE static inline dest_t apply(src_t src) {
    constexpr bool real = needs_real<dest_t, src_t>::value;
    return static_cast<dest_t>(maybe_real<real, src_t>::apply(src));
  }
};

// Partial template instantiation for casting to uint8.
// Note: Converting from negative float values to unsigned integer types is
// undefined behavior in C++, and current CPU and GPU compilers exhibit
// divergent behavior. Casting from negative float values to signed
// integer types and then to unsigned integer types is not undefined,
// however, so this cast improves the consistency of type conversions
// to uint8 across compilers.
// Further note: Type conversions across compilers still have other undefined
// and divergent behavior.
template <typename src_t>
struct static_cast_with_inter_type<uint8_t, src_t> {
  C10_HOST_DEVICE static inline uint8_t apply(src_t src) {
    constexpr bool real = needs_real<uint8_t, src_t>::value;
    return static_cast<uint8_t>(
      static_cast<int64_t>(maybe_real<real, src_t>::apply(src)));
  }
};

// Fetch a value with dynamic type src_type from ptr, and cast it to static type dest_t.
// For now, simplified version that does not handle complex and special casting to uint8
#define FETCH_AND_CAST_CASE(type, scalartype) case ScalarType::scalartype: return static_cast_with_inter_type<dest_t, type>::apply(*(const type *)ptr);
template<typename dest_t>
__device__ inline dest_t fetch_and_cast(const ScalarType src_type, const void *ptr) {
  switch (src_type) {
    AT_FORALL_SCALAR_TYPES(FETCH_AND_CAST_CASE)
    default:
      ERROR_UNSUPPORTED_CAST
  }
  return dest_t(0); // just to avoid compiler warning
}

// Cast a value with static type src_t into dynamic dest_type, and store it to ptr.
#define CAST_AND_STORE_CASE(type, scalartype) case ScalarType::scalartype: static_cast_with_inter_type<type, src_t>::apply(value); return;
template<typename src_t>
__device__ inline void cast_and_store(const ScalarType dest_type, void *ptr, src_t value) {
  switch (dest_type) {
    AT_FORALL_SCALAR_TYPES(CAST_AND_STORE_CASE)
    default:;
  }
  ERROR_UNSUPPORTED_CAST
}

}
