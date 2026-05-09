from load_data import load_default_data
from multiGPU_IVF_FLat import (
    build_ivf_flat_index,
    create_index_params,
    create_multi_gpu_resources,
    create_search_params,
    search_ivf_flat,
)
from computeL2 import get_or_compute_exact_ground_truth
from calculateRecall import calculate_recall_at_k
from IVF_flat import K, N_PROBES_SWEEP
from config import (
    DISPLAY_TOP_K,
    GROUND_TRUTH_BATCH_SIZE,
    GROUND_TRUTH_TOP_K,
    MS_PER_SECOND,
    OFFLINE_QUERY_COUNT,
    ONLINE_QUERY_COUNT,
    QUERY_LIMIT,
    SEARCH_TIMED_RUNS,
    SEARCH_WARMUP_RUNS,
)
from timing_utils import measure_synchronized_wall_time

def run_sweep_benchmark():
    dataset_ids, dataset, _, queries = load_default_data(print_info=True)
    gt_distances, gt_neighbors = get_or_compute_exact_ground_truth(
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

    index_params = create_index_params()
    
    benchmark_queries = queries[:OFFLINE_QUERY_COUNT]
    online_queries = queries[:ONLINE_QUERY_COUNT]
    num_queries = benchmark_queries.shape[0]

    print("\nRunning benchmark on IVF-Flat with multi-GPU search...")
    print("Dataset shape:", dataset.shape)
    print("Queries shape:", benchmark_queries.shape)
    print("k:", K)

    index, build_time = build_ivf_flat_index(
        dataset,
        index_params,
        resources=resources,
        sync_fn=sync_fn,
    )
    print(f"IVF-Flat index built in {build_time:.2f} seconds")

    for n_probes in N_PROBES_SWEEP:
        search_params = create_search_params(n_probes)
        print(f"\nPerforming search with n_probes={n_probes}...")
        
        offline_summary = measure_synchronized_wall_time(
            lambda: search_ivf_flat(index, benchmark_queries, search_params, K, resources=resources),
            warmup_runs=SEARCH_WARMUP_RUNS,
            timed_runs=SEARCH_TIMED_RUNS,
            sync_fn=sync_fn,
        )
        distances, neighbors = offline_summary["result"]
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

        #Compute speed numbers
        queries_per_second = num_queries / search_time
        print(f"Offline batch search median: {offline_summary['median_sec']:.4f} seconds")
        print(f"Offline batch search mean: {offline_summary['mean_sec']:.4f} seconds")
        print(f"Offline throughput median: {queries_per_second:.2f} queries/second")
        latency_per_query = search_time * MS_PER_SECOND / num_queries
        print(f"Offline latency per query median: {latency_per_query:.4f} ms")
        print(f"Online single-query median latency: {online_summary['median_sec'] * MS_PER_SECOND / ONLINE_QUERY_COUNT:.4f} ms")
        print(f"Online single-query p95 latency: {online_summary['p95_sec'] * MS_PER_SECOND / ONLINE_QUERY_COUNT:.4f} ms")
        #compute recall
        recall_at_10, total_correct, total_possible = calculate_recall_at_k(neighbors, gt_neighbors, k=K)
        print(f"Recall@10: {recall_at_10:.4f} ({total_correct} out of {total_possible} correct neighbors)")

if __name__ == "__main__":
    run_sweep_benchmark()
