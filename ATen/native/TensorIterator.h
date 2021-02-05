#pragma once

#include <vector>
#include <cstdint>
#include <c10/core/ScalarType.h>

extern std::vector<int64_t> shape;
extern std::vector<at::ScalarType> dtypes;

namespace at {

struct TensorIteratorBase {

std::vector<int64_t> &shape() {
  return ::shape;
}

at::ScalarType dtype(int64_t i) {
  return ::dtypes[i];
}

} iter;

}
