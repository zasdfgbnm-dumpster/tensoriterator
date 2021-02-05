#pragma once

float *arange(int64_t size) {
  float *buf = new float[size];
  for (int64_t i = 0; i < size; i++) {
    buf[i] = float(i);
  }
  return buf;
}