#pragma once

#include "cagra_common.hpp"

#include <cuvs/distance/distance.h>
#include <cuvs/neighbors/cagra.h>
#include <cuvs/neighbors/mg_cagra.h>

struct CagraConfig {
  int graph_degree = 32;
  int intermediate_graph_degree = 64;
  std::string build_algo = "ivf_pq";

  bool compression_enabled = true;
  int compression_pq_bits = 8;
  int compression_pq_dim = 384;

  bool enable_exact_rerank = true;
  int rerank_candidate_k = 48;

  int itopk_size = 48;
  int max_queries = 256;
  int max_iterations = 0;
  std::string algo = "auto";
  int team_size = 0;
  int search_width = 1;
  int min_iterations = 0;
  int thread_block_size = 0;
  std::string hashmap_mode = "auto";
  int hashmap_min_bitlen = 0;
  float hashmap_max_fill_rate = 0.5f;
  uint32_t num_random_samplings = 1;
  uint64_t rand_xor_mask = 0x128394;
  int64_t n_rows_per_batch = 256;
};

inline bool exact_rerank_enabled(const CagraConfig& c)
{
  const bool default_enabled =
    getenv_or("CUVS_BENCH_CAGRA_ENABLE_EXACT_RERANK", "1") != "0";
  return c.enable_exact_rerank && default_enabled;
}

inline int get_search_k(const CagraConfig& c)
{
  if (!exact_rerank_enabled(c)) return K;
  const int default_candidate_k = getenv_int_or("CUVS_BENCH_CAGRA_RERANK_CANDIDATE_K", 48);
  return std::max(K, c.rerank_candidate_k > 0 ? c.rerank_candidate_k : default_candidate_k);
}

inline cuvsDistanceType distance_from_metric(const std::string& metric)
{
  if (metric == "sqeuclidean") return L2Expanded;
  if (metric == "inner_product") return InnerProduct;
  throw std::runtime_error("Multi-GPU CAGRA supports sqeuclidean and inner_product here");
}

inline cuvsCagraSearchAlgo search_algo_from_string(const std::string& name)
{
  if (name == "auto") return AUTO;
  if (name == "single_cta") return SINGLE_CTA;
  if (name == "multi_cta") return MULTI_CTA;
  if (name == "multi_kernel") return MULTI_KERNEL;
  throw std::runtime_error("unsupported CAGRA search algo: " + name);
}

inline cuvsCagraHashMode hash_mode_from_string(const std::string& name)
{
  if (name == "auto") return AUTO_HASH;
  if (name == "hash") return HASH;
  if (name == "small") return SMALL;
  throw std::runtime_error("unsupported CAGRA hashmap mode: " + name);
}

inline cuvsMultiGpuDistributionMode distribution_mode_from_string(const std::string& mode)
{
  if (mode == "sharded") return CUVS_NEIGHBORS_MG_SHARDED;
  if (mode == "replicated") return CUVS_NEIGHBORS_MG_REPLICATED;
  throw std::runtime_error("unsupported distribution mode: " + mode);
}

inline cuvsMultiGpuReplicatedSearchMode search_mode_from_string(const std::string& mode)
{
  if (mode == "load_balancer") return CUVS_NEIGHBORS_MG_LOAD_BALANCER;
  if (mode == "round_robin") return CUVS_NEIGHBORS_MG_ROUND_ROBIN;
  throw std::runtime_error("unsupported search mode: " + mode);
}

inline cuvsMultiGpuShardedMergeMode merge_mode_from_string(const std::string& mode)
{
  if (mode == "merge_on_root_rank") return CUVS_NEIGHBORS_MG_MERGE_ON_ROOT_RANK;
  if (mode == "tree_merge") return CUVS_NEIGHBORS_MG_TREE_MERGE;
  throw std::runtime_error("unsupported merge mode: " + mode);
}

struct IndexParamsOwner {
  cuvsMultiGpuCagraIndexParams_t mg = nullptr;
  cuvsCagraCompressionParams_t compression = nullptr;

  explicit IndexParamsOwner(const CagraConfig& c)
  {
    check_cuvs(cuvsMultiGpuCagraIndexParamsCreate(&mg), "cuvsMultiGpuCagraIndexParamsCreate");
    mg->mode = distribution_mode_from_string(DISTRIBUTION_MODE);

    cuvsCagraIndexParams_t base = mg->base_params;
    base->metric = distance_from_metric(METRIC);
    base->graph_degree = static_cast<size_t>(c.graph_degree);
    base->intermediate_graph_degree = static_cast<size_t>(c.intermediate_graph_degree);

    if (c.build_algo == "ivf_pq") {
      base->build_algo = IVF_PQ;
    } else if (c.build_algo == "auto") {
      base->build_algo = AUTO_SELECT;
    } else if (c.build_algo == "nn_descent") {
      base->build_algo = NN_DESCENT;
    } else {
      throw std::runtime_error("unsupported build_algo: " + c.build_algo);
    }

    if (c.compression_enabled) {
      check_cuvs(cuvsCagraCompressionParamsCreate(&compression),
                 "cuvsCagraCompressionParamsCreate");
      compression->pq_bits = static_cast<uint32_t>(c.compression_pq_bits);
      compression->pq_dim = static_cast<uint32_t>(c.compression_pq_dim);
      base->compression = compression;
    } else {
      base->compression = nullptr;
    }
  }

  ~IndexParamsOwner()
  {
    if (compression != nullptr) {
      try {
        check_cuvs(cuvsCagraCompressionParamsDestroy(compression),
                   "cuvsCagraCompressionParamsDestroy");
      } catch (...) {
      }
    }
    if (mg != nullptr) {
      try {
        check_cuvs(cuvsMultiGpuCagraIndexParamsDestroy(mg),
                   "cuvsMultiGpuCagraIndexParamsDestroy");
      } catch (...) {
      }
    }
  }

  IndexParamsOwner(const IndexParamsOwner&) = delete;
  IndexParamsOwner& operator=(const IndexParamsOwner&) = delete;
};

struct SearchParamsOwner {
  cuvsMultiGpuCagraSearchParams_t mg = nullptr;

  explicit SearchParamsOwner(const CagraConfig& c)
  {
    const int search_k = get_search_k(c);
    if (c.itopk_size < search_k) {
      throw std::runtime_error("CAGRA itopk_size should be at least search_k");
    }

    check_cuvs(cuvsMultiGpuCagraSearchParamsCreate(&mg), "cuvsMultiGpuCagraSearchParamsCreate");

    mg->search_mode = search_mode_from_string(SEARCH_MODE);
    mg->merge_mode = merge_mode_from_string(MERGE_MODE);
    mg->n_rows_per_batch = c.n_rows_per_batch;

    cuvsCagraSearchParams_t base = mg->base_params;
    base->itopk_size = static_cast<size_t>(c.itopk_size);
    base->max_queries = static_cast<size_t>(c.max_queries);
    base->max_iterations = static_cast<size_t>(c.max_iterations);
    base->algo = search_algo_from_string(c.algo);
    base->team_size = static_cast<size_t>(c.team_size);
    base->search_width = static_cast<size_t>(c.search_width);
    base->min_iterations = static_cast<size_t>(c.min_iterations);
    base->thread_block_size = static_cast<size_t>(c.thread_block_size);
    base->hashmap_mode = hash_mode_from_string(c.hashmap_mode);
    base->hashmap_min_bitlen = static_cast<size_t>(c.hashmap_min_bitlen);
    base->hashmap_max_fill_rate = c.hashmap_max_fill_rate;
    base->num_random_samplings = c.num_random_samplings;
    base->rand_xor_mask = c.rand_xor_mask;
  }

  ~SearchParamsOwner()
  {
    if (mg != nullptr) {
      try {
        check_cuvs(cuvsMultiGpuCagraSearchParamsDestroy(mg),
                   "cuvsMultiGpuCagraSearchParamsDestroy");
      } catch (...) {
      }
    }
  }

  SearchParamsOwner(const SearchParamsOwner&) = delete;
  SearchParamsOwner& operator=(const SearchParamsOwner&) = delete;
};
