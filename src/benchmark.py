from load_data import load_default_data
from multiGPU_IVF_FLat import create_index_params, create_search_params
from computeL2 import compute_exact_ground_truth
from calculateRecall import calculate_recall_at_k
from IVF_flat import K, N_PROBES_SWEEP
from config import (
    BENCHMARK_QUERY_COUNT,
    DISPLAY_TOP_K,
    GROUND_TRUTH_BATCH_SIZE,
    GROUND_TRUTH_TOP_K,
    MS_PER_SECOND,
    QUERY_LIMIT,
)

import time
from cuvs.neighbors.mg import ivf_flat

def run_benchmark():
    dataset_ids, dataset, _, queries = load_default_data()
    gt_distances, gt_neighbors = compute_exact_ground_truth(
        dataset=dataset,
        dataset_ids=dataset_ids,
        queries=queries,
        top_k=GROUND_TRUTH_TOP_K,
        query_limit=QUERY_LIMIT,
        batch_size=GROUND_TRUTH_BATCH_SIZE,
        print_progress=True,
    )

    index_params = create_index_params()
    search_params = create_search_params(N_PROBES_SWEEP[-1])

    benchmark_queries = queries[:BENCHMARK_QUERY_COUNT]
    num_queries = benchmark_queries.shape[0]

    print("\nRunning benchmark on IVF-Flat with multi-GPU search...")
    print("Dataset shape:", dataset.shape)
    print("Queries shape:", benchmark_queries.shape)
    print("k:", K)

    build_start = time.perf_counter()
    index = ivf_flat.build(index_params, dataset)
    build_end = time.perf_counter()
    build_time = build_end - build_start
    print(f"IVF-Flat index built in {build_time:.2f} seconds")

    #Warmup search
    print("\nPerforming warmup search...")
    _ = ivf_flat.search(search_params, index, benchmark_queries, K)
    print("\nPerforming timed search...")

    search_start = time.perf_counter()
    distances, neighbors = ivf_flat.search(search_params, index, benchmark_queries, K)
    search_end = time.perf_counter()
    search_time = search_end - search_start
    print(f"IVF-Flat search completed in {search_time:.2f} seconds")
    print("Distances shape:", distances.shape)
    print("Neighbors shape:", neighbors.shape)
    print("First query top 10 neighbors:", neighbors[0, :DISPLAY_TOP_K])
    print("First query top 10 distances:", distances[0, :DISPLAY_TOP_K])

    #Compute speed numbers
    queries_per_second = num_queries / search_time
    print(f"Throughput: {queries_per_second:.2f} queries/second")
    latency_per_query = search_time * MS_PER_SECOND / num_queries
    print(f"Average latency per query: {latency_per_query:.2f} ms")

    #compute recall
    recall_at_10, total_correct, total_possible = calculate_recall_at_k(neighbors, gt_neighbors[:num_queries], k=K)
    print(f"\nRecall@10: {recall_at_10:.4f} ({total_correct} out of {total_possible} correct neighbors)")

    print("\nBenchmark complete.")

    return {
        "gt_distances": gt_distances,
        "gt_neighbors": gt_neighbors,
        "distances": distances,
        "neighbors": neighbors,
        "build_time": build_time,
        "search_time": search_time,
        "queries_per_second": queries_per_second,
        "latency_per_query": latency_per_query,
        "recall_at_10": recall_at_10,
        "total_correct": total_correct,
        "total_possible": total_possible,
    }

def main():
    run_benchmark()

if __name__ == "__main__":
    main()
