#pragma once

#include <vector>
#include <cstdint>
#include <functional>
#include <numeric>
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

int64_t ntensors() {
  return ::dtypes.size();
}

int64_t numel() {
  return std::accumulate(::shape.begin(), ::shape.end(), 1, std::multiplies<int64_t>());
}

} iter;

}
