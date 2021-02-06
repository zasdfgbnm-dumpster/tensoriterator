#pragma once

#include <vector>
#include <cstdint>
#include <functional>
#include <numeric>

extern std::vector<int64_t> shape;
extern std::vector<std::vector<int64_t>> strides;
extern std::vector<char *> data_ptrs;

namespace at {

struct TensorIteratorBase {

std::vector<int64_t> &shape() const {
  return ::shape;
}

int64_t numel() const {
  return std::accumulate(::shape.begin(), ::shape.end(), 1, std::multiplies<int64_t>());
}

int64_t ndim() const {
  return ::shape.size();
}

char *data_ptr(int64_t i) const {
  return ::data_ptrs[i];
}

std::vector<std::int64_t> &strides(int64_t i) const {
  return ::strides[i];
}

};

}
