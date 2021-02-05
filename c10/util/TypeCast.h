#pragma once

#include <c10/core/ScalarType.h>

#define ERROR_UNSUPPORTED_CAST assert(false);

// Fetch a value with dynamic type src_type from ptr, and cast it to static type dest_t.
// For now, simplified version that does not handle complex and special casting to uint8
#define FETCH_AND_CAST_CASE(type, scalartype) case ScalarType::scalartype: return static_cast<dest_t>(*(const type *)ptr);
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
#define CAST_AND_STORE_CASE(type, scalartype) case ScalarType::scalartype: *(type *)ptr = static_cast<type>(value); return;
template<typename src_t>
__device__ inline void cast_and_store(const ScalarType dest_type, void *ptr, src_t value) {
  switch (dest_type) {
    AT_FORALL_SCALAR_TYPES(CAST_AND_STORE_CASE)
    default:;
  }
  ERROR_UNSUPPORTED_CAST
}
