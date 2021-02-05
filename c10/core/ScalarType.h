#pragma once

#include <cstdint>

namespace at {

// NB: Order matters for this macro; it is relied upon in
// _promoteTypesLookup and the serialization format.
// Note, some types have ctype as void because we don't support them in codegen
#define AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_QINTS(_)                       \
  _(uint8_t, Byte)       /* 0 */                                               \
  _(int8_t, Char)        /* 1 */                                               \
  _(int16_t, Short)      /* 2 */                                               \
  _(int, Int)            /* 3 */                                               \
  _(int64_t, Long)       /* 4 */                                               \
  _(void, Half)          /* 5 */                                               \
  _(float, Float)        /* 6 */                                               \
  _(double, Double)      /* 7 */                                               \
  _(void, ComplexHalf)   /* 8 */                                               \
  _(void, ComplexFloat)  /* 9 */                                               \
  _(void, ComplexDouble) /* 10 */                                              \
  _(bool, Bool)          /* 11 */                                              \
  _(void, QInt8)         /* 12 */                                              \
  _(void, QUInt8)        /* 13 */                                              \
  _(void, QInt32)        /* 14 */                                              \
  _(void, BFloat16)      /* 15 */                                              \
  _(void, QUInt4x2)      /* 16 */

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

}