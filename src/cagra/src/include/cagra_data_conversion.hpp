#pragma once

#include "cagra_common.hpp"

template <typename T>
inline std::vector<T> convert_float_dataset(const std::vector<float>& src)
{
  std::vector<T> dst(src.size());
  for (size_t i = 0; i < src.size(); ++i) dst[i] = static_cast<T>(src[i]);
  return dst;
}

template <>
inline std::vector<__half> convert_float_dataset<__half>(const std::vector<float>& src)
{
  std::vector<__half> dst(src.size());
  #pragma omp parallel for schedule(static)
  for (size_t i = 0; i < src.size(); ++i) dst[i] = __float2half(src[i]);
  return dst;
}

template <typename T>
inline PinnedHostBuffer<T> convert_float_dataset_pinned(const std::vector<float>& src)
{
  PinnedHostBuffer<T> dst(src.size());
  #pragma omp parallel for schedule(static)
  for (int64_t i = 0; i < static_cast<int64_t>(src.size()); ++i) {
    dst[static_cast<size_t>(i)] = static_cast<T>(src[static_cast<size_t>(i)]);
  }
  return dst;
}

template <>
inline PinnedHostBuffer<__half> convert_float_dataset_pinned<__half>(
  const std::vector<float>& src)
{
  PinnedHostBuffer<__half> dst(src.size());
  #pragma omp parallel for schedule(static)
  for (int64_t i = 0; i < static_cast<int64_t>(src.size()); ++i) {
    dst[static_cast<size_t>(i)] = __float2half(src[static_cast<size_t>(i)]);
  }
  return dst;
}


template <typename T>
inline PinnedHostBuffer<T> make_pinned_slice(const std::vector<T>& src, int64_t row_start, int64_t rows, int64_t cols)
{
  PinnedHostBuffer<T> out(static_cast<size_t>(rows * cols));
  const size_t offset = static_cast<size_t>(row_start * cols);
  std::copy(src.begin() + offset, src.begin() + offset + out.count, out.data());
  return out;
}

inline PinnedHostBuffer<float> make_pinned_float_slice(
  const std::vector<float>& src,
  int64_t row_start,
  int64_t rows,
  int64_t cols)
{
  PinnedHostBuffer<float> out(static_cast<size_t>(rows * cols));
  const size_t offset = static_cast<size_t>(row_start * cols);
  std::copy(src.begin() + offset, src.begin() + offset + out.count, out.data());
  return out;
}
