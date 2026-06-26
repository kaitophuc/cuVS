#pragma once

#include "cagra_common.hpp"

#include <dlpack/dlpack.h>

DLDataType dl_float32();
DLDataType dl_float16();
DLDataType dl_int64();
DLDataType dl_int32();

template <typename T>
DLDataType dtype_for_dlpack();

template <>
inline DLDataType dtype_for_dlpack<float>()
{
  return dl_float32();
}

template <>
inline DLDataType dtype_for_dlpack<__half>()
{
  return dl_float16();
}

struct DlpackMatrix {
  DLManagedTensor managed{};
  std::vector<int64_t> shape;

  DlpackMatrix(void* data, int64_t rows, int64_t cols, DLDataType dtype);
  DLManagedTensor* get();
};

struct DlpackVector {
  DLManagedTensor managed{};
  std::vector<int64_t> shape;

  DlpackVector(void* data, int64_t rows, DLDataType dtype);
  DLManagedTensor* get();
};
