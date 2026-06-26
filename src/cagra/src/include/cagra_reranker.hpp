#pragma once

#include "cagra_common.hpp"
#include "cagra_data.hpp"

class MultiGpuExactReranker {
public:
  MultiGpuExactReranker(
    const LoadedData& data,
    int64_t final_k,
    int64_t candidate_k,
    int64_t batch_size,
    const float* pinned_float_dataset,
    const __half* pinned_half_dataset,
    const std::optional<std::vector<int>>& device_ids,
    const std::string& storage_dtype);
  ~MultiGpuExactReranker();

  MultiGpuExactReranker(const MultiGpuExactReranker&) = delete;
  MultiGpuExactReranker& operator=(const MultiGpuExactReranker&) = delete;

  void rerank(
    const float* queries_ptr,
    const int64_t* candidates_ptr,
    int64_t n_queries,
    float* out_distances_ptr,
    int64_t* out_neighbors_ptr);

  std::string mode() const;

private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

std::unique_ptr<MultiGpuExactReranker> create_cagra_exact_reranker(
  const LoadedData& data,
  int candidate_k,
  int batch_size,
  const float* pinned_float_dataset,
  const __half* pinned_half_dataset,
  const std::optional<std::vector<int>>& device_ids,
  const std::string& requested_storage_dtype,
  std::string* active_storage_dtype);
