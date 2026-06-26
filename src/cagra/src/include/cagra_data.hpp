#pragma once

#include "cagra_common.hpp"

struct LoadedData {
  std::vector<int64_t> dataset_ids;
  std::vector<float> dataset;
  std::vector<float> queries;
  int64_t n_rows = 0;
  int64_t dim = 0;
  int64_t n_queries = 0;
};

struct GroundTruth {
  std::vector<int64_t> neighbors;
  int64_t rows = 0;
  int64_t cols = 0;
};

fs::path project_root();
LoadedData load_default_data_from_npy_cache();
GroundTruth load_required_ground_truth();
