#pragma once

#include <cstdint>

namespace c10 {

// NB: Order matters for this macro; it is relied upon in
// _promoteTypesLookup and the serialization format.
// Note, some types have ctype as void because we don't support them in codegen
#define AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_QINTS(_)                       \
  _(uint8_t, Byte)       /* 0 */                                               \
  _(int8_t, Char)        /* 1 */                                               \
  _(int16_t, Short)      /* 2 */                                               \
  _(int, Int)            /* 3 */                                               \
  _(int64_t, Long)       /* 4 */                                               \
  _(float, Float)        /* 6 */                                               \
  _(double, Double)      /* 7 */                                               \
  _(bool, Bool)

#define AT_FORALL_SCALAR_TYPES(_)                                              \
  _(uint8_t, Byte)                                                             \
  _(int8_t, Char)                                                              \
  _(int16_t, Short)                                                            \
  _(int, Int)                                                                  \
  _(int64_t, Long)                                                             \
  _(float, Float)                                                              \
  _(double, Double)

enum class ScalarType : int8_t {
#define DEFINE_ENUM(_1, n) n,
  AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_QINTS(DEFINE_ENUM)
#undef DEFINE_ENUM
      Undefined,
  NumOptions
};

static inline size_t elementSize(ScalarType t) {
#define CASE_ELEMENTSIZE_CASE(ctype, name) \
  case ScalarType::name:                   \
    return sizeof(ctype);

  switch (t) {
    AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_QINTS(CASE_ELEMENTSIZE_CASE)
    default:
      AT_ERROR("Unknown ScalarType");
  }
  return 0;
#undef CASE_ELEMENTSIZE_CASE
}

template <typename T>
struct CppTypeToScalarType;

#define SPECIALIZE_CppTypeToScalarType(cpp_type, scalar_type) \
  template<>                                                  \
  struct CppTypeToScalarType<cpp_type>:                       \
    std::integral_constant<c10::ScalarType,                   \
                           c10::ScalarType::scalar_type>      \
  {};

AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_QINTS(SPECIALIZE_CppTypeToScalarType);

#undef SPECIALIZE_CppTypeToScalarType

}

namespace at {
using namespace c10;
}

using namespace at;
