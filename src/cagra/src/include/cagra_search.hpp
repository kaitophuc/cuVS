#pragma once

#include "cagra_common.hpp"
#include "cagra_data.hpp"
#include "cagra_dlpack.hpp"
#include "cagra_params.hpp"
#include "cagra_reranker.hpp"
#include "cagra_resources.hpp"

struct SearchOutput {
  std::vector<float> distances;
  std::vector<int64_t> neighbors;
  int64_t rows = 0;
  int64_t cols = 0;
};

template <typename T>
struct SearchRunner {
  const MultiGpuResources& resources;
  cuvsMultiGpuCagraIndex_t index;
  const SearchParamsOwner& params;
  MatrixViewHost<const T> queries;
  MatrixViewHost<const float> rerank_queries;
  const LoadedData& data;
  int final_k = K;
  int search_k = K;
  bool exact_rerank = true;
  MultiGpuExactReranker* reranker = nullptr;

  PinnedHostBuffer<int64_t> row_neighbors;
  PinnedHostBuffer<float> candidate_distances;

  SearchOutput last_output;

  SearchRunner(
    const MultiGpuResources& resources_,
    cuvsMultiGpuCagraIndex_t index_,
    const SearchParamsOwner& params_,
    MatrixViewHost<const T> queries_,
    MatrixViewHost<const float> rerank_queries_,
    const LoadedData& data_,
    int final_k_,
    int search_k_,
    bool exact_rerank_,
    MultiGpuExactReranker* reranker_)
    : resources(resources_),
      index(index_),
      params(params_),
      queries(queries_),
      rerank_queries(rerank_queries_),
      data(data_),
      final_k(final_k_),
      search_k(search_k_),
      exact_rerank(exact_rerank_),
      reranker(reranker_),
      row_neighbors(static_cast<size_t>(queries_.rows * search_k_)),
      candidate_distances(static_cast<size_t>(queries_.rows * search_k_))
  {
    last_output.rows = queries.rows;
    last_output.cols = final_k;
    last_output.distances.resize(static_cast<size_t>(queries.rows * final_k));
    last_output.neighbors.resize(static_cast<size_t>(queries.rows * final_k));
  }

  void run()
  {
    DlpackMatrix query_tensor(
      const_cast<T*>(queries.data), queries.rows, queries.cols, dtype_for_dlpack<T>());
    DlpackMatrix neighbor_tensor(row_neighbors.data(), queries.rows, search_k, dl_int64());
    DlpackMatrix distance_tensor(candidate_distances.data(), queries.rows, search_k, dl_float32());

    check_cuvs(cuvsMultiGpuCagraSearch(resources.handle,
                                       params.mg,
                                       index,
                                       query_tensor.get(),
                                       neighbor_tensor.get(),
                                       distance_tensor.get()),
               "cuvsMultiGpuCagraSearch");

    resources.sync();

    if (exact_rerank) {
      if (reranker == nullptr) {
        throw std::runtime_error("exact CAGRA rerank requested without a multi-GPU reranker");
      }
      reranker->rerank(
        rerank_queries.data,
        row_neighbors.data(),
        queries.rows,
        last_output.distances.data(),
        last_output.neighbors.data());
      return;
    }

    for (int64_t q = 0; q < queries.rows; ++q) {
      for (int k = 0; k < final_k; ++k) {
        const int64_t row = row_neighbors[static_cast<size_t>(q * search_k + k)];
        if (row < 0 || row >= data.n_rows) {
          throw std::runtime_error("CAGRA returned row neighbor outside dataset range");
        }

        last_output.distances[static_cast<size_t>(q * final_k + k)] =
          candidate_distances[static_cast<size_t>(q * search_k + k)];
        last_output.neighbors[static_cast<size_t>(q * final_k + k)] =
          data.dataset_ids[static_cast<size_t>(row)];
      }
    }
  }
};
