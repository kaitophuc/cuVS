#include "cagra_runner.hpp"

#include "cagra_common.hpp"
#include "cagra_data.hpp"
#include "cagra_data_conversion.hpp"
#include "cagra_dlpack.hpp"
#include "cagra_params.hpp"
#include "cagra_reranker.hpp"
#include "cagra_resources.hpp"
#include "cagra_results.hpp"
#include "cagra_search.hpp"
#include "cagra_timing.hpp"

#include <cuvs/neighbors/mg_cagra.h>

template <typename T>
void run_cagra_optimized_typed(
  const CagraConfig& config,
  const fs::path& output_path)
{
  LoadedData data = load_default_data_from_npy_cache();
  GroundTruth ground_truth = load_required_ground_truth();

  PinnedHostBuffer<T> index_dataset = convert_float_dataset_pinned<T>(data.dataset);
  const std::vector<T> search_queries = convert_float_dataset<T>(data.queries);

  const float* pinned_float_dataset = nullptr;
  const __half* pinned_half_dataset = nullptr;
  if constexpr (std::is_same_v<T, float>) {
    pinned_float_dataset = index_dataset.data();
  } else {
    pinned_half_dataset = index_dataset.data();
  }

  const auto device_ids = optional_int_list_from_env("CUVS_BENCH_CAGRA_DEVICE_IDS");
  auto rerank_device_ids = optional_int_list_from_env("CUVS_BENCH_CAGRA_RERANK_DEVICE_IDS");
  if (!rerank_device_ids.has_value()) rerank_device_ids = device_ids;
  MultiGpuResources resources(device_ids);

  const bool exact_rerank = exact_rerank_enabled(config);
  const int search_k = get_search_k(config);

  const int64_t offline_rows = std::min<int64_t>(OFFLINE_QUERY_COUNT, data.n_queries);
  const int64_t online_rows = std::min<int64_t>(ONLINE_QUERY_COUNT, data.n_queries);

  PinnedHostBuffer<T> benchmark_queries =
    make_pinned_slice(search_queries, 0, offline_rows, data.dim);
  PinnedHostBuffer<T> online_queries =
    make_pinned_slice(search_queries, 0, online_rows, data.dim);

  PinnedHostBuffer<float> benchmark_rerank_queries =
    make_pinned_float_slice(data.queries, 0, offline_rows, data.dim);
  PinnedHostBuffer<float> online_rerank_queries =
    make_pinned_float_slice(data.queries, 0, online_rows, data.dim);

  std::cout << "\nRunning CAGRA benchmark\n";
  std::cout << "Dataset shape: (" << data.n_rows << ", " << data.dim << ")\n";
  std::cout << "Dataset dtype: " << getenv_or("CUVS_BENCH_CAGRA_DATASET_DTYPE", "float16") << "\n";
  std::cout << "Queries shape: (" << offline_rows << ", " << data.dim << ")\n";
  std::cout << "Queries dtype: " << getenv_or("CUVS_BENCH_CAGRA_QUERY_DTYPE", "float16") << "\n";
  std::cout << "k: " << K << "\n";
  std::cout << "distribution mode: " << DISTRIBUTION_MODE << "\n";
  std::cout << "search mode: " << SEARCH_MODE << "\n";
  std::cout << "merge mode: " << MERGE_MODE << "\n";
  if (device_ids.has_value()) {
    std::cout << "device ids:";
    for (int id : *device_ids) std::cout << ' ' << id;
    std::cout << "\n";
  } else {
    std::cout << "device ids: all visible\n";
  }

  const std::string rerank_backend =
    exact_rerank ? getenv_or("CUVS_BENCH_CAGRA_RERANK_BACKEND", "multi_gpu") : "none";
  if (exact_rerank && rerank_backend != "multi_gpu") {
    throw std::runtime_error(
      "CUVS_BENCH_CAGRA_RERANK_BACKEND must be 'multi_gpu'. "
      "Only multi-GPU exact rerank is supported by this runner.");
  }
  const int rerank_batch_size = getenv_int_or("CUVS_BENCH_CAGRA_RERANK_BATCH_SIZE", 512);
  const std::string requested_rerank_storage_dtype =
    getenv_or("CUVS_BENCH_CAGRA_RERANK_STORAGE_DTYPE", "float16");

  std::cout << "exact rerank: " << (exact_rerank ? "enabled" : "disabled") << "\n";
  std::cout << "rerank backend: " << rerank_backend << "\n";
  if (exact_rerank) {
    std::cout << "rerank storage dtype: " << requested_rerank_storage_dtype << "\n";
    std::cout << "rerank batch size: " << rerank_batch_size << "\n";
    if (rerank_device_ids.has_value()) {
      std::cout << "rerank device ids:";
      for (int id : *rerank_device_ids) std::cout << ' ' << id;
      std::cout << "\n";
    } else {
      std::cout << "rerank device ids: all visible\n";
    }
  }
  std::cout << "pinned host search I/O: enabled\n";

  std::vector<ResultRow> results;

  std::cout << "\nBuilding CAGRA index with "
            << "graph_degree=" << config.graph_degree << ", "
            << "intermediate_graph_degree=" << config.intermediate_graph_degree << ", "
            << "build_algo=" << config.build_algo << "...\n";

  IndexParamsOwner index_params(config);
  MultiGpuCagraIndex index;

  DlpackMatrix dataset_tensor(
    const_cast<T*>(index_dataset.data()), data.n_rows, data.dim, dtype_for_dlpack<T>());

  resources.sync();
  auto build_start = std::chrono::steady_clock::now();

  check_cuvs(cuvsMultiGpuCagraBuild(resources.handle,
                                    index_params.mg,
                                    dataset_tensor.get(),
                                    index.index),
             "cuvsMultiGpuCagraBuild");

  resources.sync();
  auto build_end = std::chrono::steady_clock::now();
  const double build_time = std::chrono::duration<double>(build_end - build_start).count();

  std::cout << "CAGRA index built in " << std::fixed << std::setprecision(2)
            << build_time << " seconds\n";

  std::unique_ptr<MultiGpuExactReranker> reranker;
  std::string reranker_storage_dtype = "none";

  std::cout << "\nitopk_size=" << config.itopk_size << ", "
            << "search_k=" << search_k << ", "
            << "search_width=" << config.search_width << ", "
            << "max_iterations=" << config.max_iterations << "\n";

  if (exact_rerank) {
    std::cout << "Creating exact CAGRA reranker "
              << "(candidate_k=" << search_k
              << ", storage=" << requested_rerank_storage_dtype << ")...\n";
    reranker = create_cagra_exact_reranker(
      data,
      search_k,
      rerank_batch_size,
      pinned_float_dataset,
      pinned_half_dataset,
      rerank_device_ids,
      requested_rerank_storage_dtype,
      &reranker_storage_dtype);
    std::cout << "Exact reranker mode: " << reranker->mode()
              << " (storage=" << reranker_storage_dtype << ")\n";
  }

  SearchParamsOwner search_params(config);

  MatrixViewHost<const T> offline_query_view{
    benchmark_queries.data(), offline_rows, data.dim};
  MatrixViewHost<const T> online_query_view{
    online_queries.data(), online_rows, data.dim};
  MatrixViewHost<const float> offline_rerank_query_view{
    benchmark_rerank_queries.data(), offline_rows, data.dim};
  MatrixViewHost<const float> online_rerank_query_view{
    online_rerank_queries.data(), online_rows, data.dim};

  SearchRunner<T> offline_search(
    resources,
    index.index,
    search_params,
    offline_query_view,
    offline_rerank_query_view,
    data,
    K,
    search_k,
    exact_rerank,
    reranker.get());

  SearchRunner<T> online_search(
    resources,
    index.index,
    search_params,
    online_query_view,
    online_rerank_query_view,
    data,
    K,
    search_k,
    exact_rerank,
    reranker.get());

  TimingSummary offline_summary = measure_synchronized_wall_time(
    [&]() { offline_search.run(); },
    SEARCH_WARMUP_RUNS,
    SEARCH_TIMED_RUNS,
    resources);

  const SearchOutput offline_output = offline_search.last_output;
  const double search_time = offline_summary.median_sec;

  TimingSummary online_summary = measure_synchronized_wall_time(
    [&]() { online_search.run(); },
    SEARCH_WARMUP_RUNS,
    SEARCH_TIMED_RUNS,
    resources);

  const double queries_per_second = static_cast<double>(offline_rows) / search_time;
  const double latency_per_query = search_time * MS_PER_SECOND / static_cast<double>(offline_rows);

  RecallResult recall = calculate_recall_at_k(offline_output, ground_truth, K);

  std::cout << "First query top neighbors:";
  for (int i = 0; i < DISPLAY_TOP_K; ++i) {
    std::cout << ' ' << offline_output.neighbors[static_cast<size_t>(i)];
  }
  std::cout << "\n";

  std::cout << "First query top distances:";
  for (int i = 0; i < DISPLAY_TOP_K; ++i) {
    std::cout << ' ' << offline_output.distances[static_cast<size_t>(i)];
  }
  std::cout << "\n";

  std::cout << std::fixed << std::setprecision(4);
  std::cout << "Offline median: " << search_time << " seconds\n";
  std::cout << "Throughput: " << std::setprecision(2) << queries_per_second
            << " queries/second\n";
  std::cout << "Latency per query: " << std::setprecision(4)
            << latency_per_query << " ms\n";
  std::cout << "Recall@10: " << recall.recall << " ("
            << recall.total_correct << "/" << recall.total_possible << ")\n";

  ResultRow row;
  row.config = config;
  row.search_k = search_k;
  row.exact_rerank = exact_rerank;
  row.rerank_backend = exact_rerank ? rerank_backend : "none";
  row.rerank_storage_dtype = exact_rerank ? reranker_storage_dtype : "none";
  row.build_time = build_time;
  row.search_time = search_time;
  row.online_summary = online_summary;
  row.queries_per_second = queries_per_second;
  row.latency_per_query = latency_per_query;
  row.recall_at_10 = recall.recall;
  row.total_correct = recall.total_correct;
  row.total_possible = recall.total_possible;
  results.push_back(std::move(row));

  resources.sync();
  write_results_csv(results, output_path);
  std::cout << "\nSaved CAGRA results to: " << output_path << "\n";
}


void run_cagra_optimized()
{
  CagraConfig optimized;

  const std::string dataset_dtype =
    getenv_or("CUVS_BENCH_CAGRA_DATASET_DTYPE", "float16");
  const std::string query_dtype =
    getenv_or("CUVS_BENCH_CAGRA_QUERY_DTYPE", dataset_dtype);

  if (dataset_dtype != query_dtype) {
    throw std::runtime_error(
      "This standalone C++ translation expects CAGRA dataset/query dtype to match. "
      "The Python binding may dispatch more flexibly.");
  }

  fs::path output_path = project_root() / "results" / "cagra_optimized_results.csv";

  if (dataset_dtype == "float16") {
    run_cagra_optimized_typed<__half>(optimized, output_path);
  } else if (dataset_dtype == "float32") {
    run_cagra_optimized_typed<float>(optimized, output_path);
  } else {
    throw std::runtime_error("Only float16 and float32 are implemented in this C++ translation");
  }
}
