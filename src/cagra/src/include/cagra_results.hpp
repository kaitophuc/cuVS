#pragma once

#include "cagra_common.hpp"
#include "cagra_data.hpp"
#include "cagra_params.hpp"
#include "cagra_search.hpp"
#include "cagra_timing.hpp"

struct RecallResult {
  double recall = 0.0;
  int64_t total_correct = 0;
  int64_t total_possible = 0;
};

RecallResult calculate_recall_at_k(
  const SearchOutput& retrieved,
  const GroundTruth& ground_truth,
  int k);

struct ResultRow {
  CagraConfig config;
  int search_k = 0;
  bool exact_rerank = false;
  std::string rerank_backend = "none";
  std::string rerank_storage_dtype = "none";
  double build_time = 0.0;
  double search_time = 0.0;
  TimingSummary online_summary;
  double queries_per_second = 0.0;
  double latency_per_query = 0.0;
  double recall_at_10 = 0.0;
  int64_t total_correct = 0;
  int64_t total_possible = 0;
};

void write_results_csv(const std::vector<ResultRow>& results, const fs::path& output_path);
