import gc

from load_data import load_default_data
from multi_gpu_ivf_flat import (
    build_ivf_flat_index,
    create_index_params,
    create_multi_gpu_resources,
    create_search_params,
    search_ivf_flat,
)
from ground_truth import get_or_compute_exact_ground_truth
from recall import calculate_recall_at_k
from config import (
    DISPLAY_TOP_K,
    GROUND_TRUTH_BATCH_SIZE,
    GROUND_TRUTH_TOP_K,
    K,
    MS_PER_SECOND,
    N_LISTS_SWEEP,
    N_PROBES_SWEEP,
    OFFLINE_QUERY_COUNT,
    ONLINE_QUERY_COUNT,
    QUERY_LIMIT,
    SEARCH_TIMED_RUNS,
    SEARCH_WARMUP_RUNS,
)
from timing_utils import measure_synchronized_wall_time


def drop_timing_result(summary):
    result = summary.pop("result", None)
    return result


def run_sweep_benchmark():
    """Run the IVF-Flat parameter sweep and return per-configuration metrics."""
    dataset_ids, dataset, _, queries = load_default_data(print_info=True)
    _, gt_neighbors = get_or_compute_exact_ground_truth(
        dataset=dataset,
        dataset_ids=dataset_ids,
        queries=queries,
        top_k=GROUND_TRUTH_TOP_K,
        query_limit=QUERY_LIMIT,
        batch_size=GROUND_TRUTH_BATCH_SIZE,
        print_info=True,
    )

    resources = create_multi_gpu_resources()
    sync_fn = resources.sync

    benchmark_queries = queries[:OFFLINE_QUERY_COUNT]
    online_queries = queries[:ONLINE_QUERY_COUNT]
    num_queries = benchmark_queries.shape[0]

    print("\nRunning benchmark on IVF-Flat with multi-GPU search...")
    print("Dataset shape:", dataset.shape)
    print("Queries shape:", benchmark_queries.shape)
    print("k:", K)
    print("n_lists sweep:", N_LISTS_SWEEP)
    print("n_probes sweep:", N_PROBES_SWEEP)

    results = []
    total_configs = len(N_LISTS_SWEEP) * len(N_PROBES_SWEEP)
    config_number = 0

    for n_lists in N_LISTS_SWEEP:
        index_params = create_index_params(n_lists=n_lists)
        print(f"\nBuilding IVF-Flat index with n_lists={n_lists}...")
        index, build_time = build_ivf_flat_index(
            dataset,
            index_params,
            resources=resources,
            sync_fn=sync_fn,
        )
        print(f"IVF-Flat index built in {build_time:.2f} seconds")

        try:
            for n_probes in N_PROBES_SWEEP:
                config_number += 1
                search_params = create_search_params(n_probes)
                print(
                    f"\n[{config_number}/{total_configs}] "
                    f"Performing search with n_lists={n_lists}, n_probes={n_probes}..."
                )

                offline_summary = measure_synchronized_wall_time(
                    lambda: search_ivf_flat(index, benchmark_queries, search_params, K, resources=resources),
                    warmup_runs=SEARCH_WARMUP_RUNS,
                    timed_runs=SEARCH_TIMED_RUNS,
                    sync_fn=sync_fn,
                )
                distances, neighbors = drop_timing_result(offline_summary)
                search_time = offline_summary["median_sec"]

                print("Distances shape:", distances.shape)
                print("Neighbors shape:", neighbors.shape)
                print("First query top 10 neighbors:", neighbors[0, :DISPLAY_TOP_K])
                print("First query top 10 distances:", distances[0, :DISPLAY_TOP_K])

                online_summary = measure_synchronized_wall_time(
                    lambda: search_ivf_flat(index, online_queries, search_params, K, resources=resources),
                    warmup_runs=SEARCH_WARMUP_RUNS,
                    timed_runs=SEARCH_TIMED_RUNS,
                    sync_fn=sync_fn,
                )
                drop_timing_result(online_summary)

                # Compute speed numbers.
                queries_per_second = num_queries / search_time
                print(f"Offline batch search median: {offline_summary['median_sec']:.4f} seconds")
                print(f"Offline batch search mean: {offline_summary['mean_sec']:.4f} seconds")
                print(f"Offline throughput median: {queries_per_second:.2f} queries/second")
                latency_per_query = search_time * MS_PER_SECOND / num_queries
                print(f"Offline latency per query median: {latency_per_query:.4f} ms")
                print(f"Online single-query median latency: {online_summary['median_sec'] * MS_PER_SECOND / ONLINE_QUERY_COUNT:.4f} ms")
                print(f"Online single-query p95 latency: {online_summary['p95_sec'] * MS_PER_SECOND / ONLINE_QUERY_COUNT:.4f} ms")
                # Compute recall.
                recall_at_10, total_correct, total_possible = calculate_recall_at_k(
                    neighbors,
                    gt_neighbors[:num_queries],
                    k=K,
                )
                print(f"Recall@10: {recall_at_10:.4f} ({total_correct} out of {total_possible} correct neighbors)")

                results.append(
                    {
                        "n_lists": n_lists,
                        "n_probes": n_probes,
                        "build_time": build_time,
                        "search_time": search_time,
                        "offline_summary": offline_summary,
                        "online_summary": online_summary,
                        "queries_per_second": queries_per_second,
                        "latency_per_query": latency_per_query,
                        "recall_at_10": recall_at_10,
                        "total_correct": total_correct,
                        "total_possible": total_possible,
                    }
                )

                del distances, neighbors
                gc.collect()
                sync_fn()
        finally:
            sync_fn()
            del index
            gc.collect()
            sync_fn()

    return results

if __name__ == "__main__":
    run_sweep_benchmark()
