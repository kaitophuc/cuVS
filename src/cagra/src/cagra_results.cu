#include "cagra_results.hpp"

RecallResult calculate_recall_at_k(
  const SearchOutput& retrieved,
  const GroundTruth& ground_truth,
  int k)
{
  RecallResult r;
  r.total_possible = retrieved.rows * k;

  for (int64_t q = 0; q < retrieved.rows; ++q) {
    for (int i = 0; i < k; ++i) {
      const int64_t got = retrieved.neighbors[static_cast<size_t>(q * retrieved.cols + i)];

      for (int j = 0; j < k; ++j) {
        const int64_t expected = ground_truth.neighbors[static_cast<size_t>(q * ground_truth.cols + j)];
        if (got == expected) {
          r.total_correct += 1;
          break;
        }
      }
    }
  }

  r.recall = static_cast<double>(r.total_correct) / static_cast<double>(r.total_possible);
  return r;
}

void write_results_csv(const std::vector<ResultRow>& results, const fs::path& output_path)
{
  fs::create_directories(output_path.parent_path());
  std::ofstream out(output_path);
  if (!out) throw std::runtime_error("failed to open output CSV: " + output_path.string());

  out << "graph_degree,intermediate_graph_degree,build_algo,"
      << "compression_enabled,compression_pq_bits,compression_pq_dim,"
      << "enable_exact_rerank,rerank_candidate_k,search_k,exact_rerank,"
      << "rerank_backend,rerank_storage_dtype,itopk_size,max_queries,"
      << "max_iterations,algo,team_size,search_width,min_iterations,"
      << "thread_block_size,hashmap_mode,hashmap_min_bitlen,"
      << "hashmap_max_fill_rate,num_random_samplings,rand_xor_mask,"
      << "n_rows_per_batch,build_time,search_time,queries_per_second,"
      << "latency_per_query,online_median_latency_ms,online_p95_latency_ms,"
      << "recall_at_10,total_correct,total_possible\n";

  out << std::setprecision(10);

  for (const ResultRow& r : results) {
    const CagraConfig& c = r.config;
    const double online_median_ms =
      r.online_summary.median_sec * MS_PER_SECOND / ONLINE_QUERY_COUNT;
    const double online_p95_ms =
      r.online_summary.p95_sec * MS_PER_SECOND / ONLINE_QUERY_COUNT;

    out << c.graph_degree << ','
        << c.intermediate_graph_degree << ','
        << c.build_algo << ','
        << (c.compression_enabled ? "True" : "False") << ','
        << c.compression_pq_bits << ','
        << c.compression_pq_dim << ','
        << (c.enable_exact_rerank ? "True" : "False") << ','
        << c.rerank_candidate_k << ','
        << r.search_k << ','
        << (r.exact_rerank ? "True" : "False") << ','
        << r.rerank_backend << ','
        << r.rerank_storage_dtype << ','
        << c.itopk_size << ','
        << c.max_queries << ','
        << c.max_iterations << ','
        << c.algo << ','
        << c.team_size << ','
        << c.search_width << ','
        << c.min_iterations << ','
        << c.thread_block_size << ','
        << c.hashmap_mode << ','
        << c.hashmap_min_bitlen << ','
        << c.hashmap_max_fill_rate << ','
        << c.num_random_samplings << ','
        << c.rand_xor_mask << ','
        << c.n_rows_per_batch << ','
        << r.build_time << ','
        << r.search_time << ','
        << r.queries_per_second << ','
        << r.latency_per_query << ','
        << online_median_ms << ','
        << online_p95_ms << ','
        << r.recall_at_10 << ','
        << r.total_correct << ','
        << r.total_possible << '\n';
  }
}
