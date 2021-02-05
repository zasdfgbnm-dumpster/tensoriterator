#pragma once

#include <vector>
#include <cstdint>
#include <functional>
#include <numeric>
#include <c10/core/ScalarType.h>

extern std::vector<int64_t> shape;
extern std::vector<at::ScalarType> dtypes;
extern std::vector<char *> data_ptrs;
extern bool is_contiguous;
extern int64_t noutput;

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

int64_t ndim() {
  return ::shape.size();
}

bool can_use_32bit_indexing() {
  return true;
}

std::array<TensorIteratorBase, 1> with_32bit_indexing() {
  return {*this};
}

char *data_ptr(int64_t i) {
  return ::data_ptrs[i];
}

bool is_contiguous() {
  return ::is_contiguous;
}

at::ScalarType input_dtype(int64_t i) {
  return ::dtypes[i + ::noutput];
}

} iter;

}
