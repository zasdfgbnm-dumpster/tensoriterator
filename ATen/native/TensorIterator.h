#pragma once

#include <vector>
#include <cstdint>
#include <functional>
#include <numeric>
#include <c10/core/ScalarType.h>
#include <array>

extern std::vector<int64_t> shape;
extern std::vector<std::vector<int64_t>> strides;
extern std::vector<at::ScalarType> dtypes;
extern std::vector<char *> data_ptrs;
extern bool is_contiguous;
extern int64_t noutputs;

namespace at {

struct TensorIteratorBase {

std::vector<int64_t> &shape() const {
  return ::shape;
}

at::ScalarType dtype(int64_t i) const {
  return ::dtypes[i];
}

int64_t ntensors() const {
  return ::dtypes.size();
}

int64_t numel() const {
  return std::accumulate(::shape.begin(), ::shape.end(), 1, std::multiplies<int64_t>());
}

int64_t ndim() const {
  return ::shape.size();
}

bool can_use_32bit_indexing() const {
  return true;
}

std::array<TensorIteratorBase, 1> with_32bit_indexing() const {
  return {*this};
}

char *data_ptr(int64_t i) const {
  return ::data_ptrs[i];
}

bool is_contiguous() const {
  return ::is_contiguous;
}

at::ScalarType input_dtype(int64_t i) const {
  return ::dtypes[i + ::noutputs];
}

int64_t noutputs() const {
  return ::noutputs;
}

int64_t element_size(int64_t i) const {
  return c10::elementSize(::dtypes[i]);
}

std::vector<std::int64_t> &strides(int64_t i) const {
  return ::strides[i];
}

bool is_cpu_scalar(int64_t i) const {
  return false;
}

template<typename T>
T scalar_value(int64_t i) {
  throw "wrong!";
}

void remove_operand(int64_t i) {
  throw "wrong!";
}

};

}
