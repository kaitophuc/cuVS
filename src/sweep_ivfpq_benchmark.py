import csv
import gc
import numpy as np

import nvtx

from recall import calculate_recall_at_k
from ground_truth import get_or_compute_exact_ground_truth
from config import (
    DISPLAY_TOP_K,
    GROUND_TRUTH_BATCH_SIZE,
    GROUND_TRUTH_TOP_K,
    IVFPQ_DATASET_DTYPE,
    IVFPQ_N_LISTS_SWEEP,
    IVFPQ_N_PROBES_SWEEP,
    IVFPQ_PQ_BITS_SWEEP,
    IVFPQ_PQ_DIM_SWEEP,
    IVFPQ_QUERY_DTYPE,
    K,
    MS_PER_SECOND,
    OFFLINE_QUERY_COUNT,
    ONLINE_QUERY_COUNT,
    QUERY_LIMIT,
    SEARCH_TIMED_RUNS,
    SEARCH_WARMUP_RUNS,
    IVFPQ_ENABLE_EXACT_RERANK,
    IVFPQ_RERANK_BACKEND,
    IVFPQ_RERANK_BATCH_SIZE,
    IVFPQ_RERANK_CANDIDATE_K,
    RESULTS_DIR,
)
from load_data import load_default_data
from multi_gpu_ivf_pq import (
    build_ivf_pq_index,
    create_index_params,
    create_multi_gpu_resources,
    create_search_params,
    dtype_from_config,
    search_ivf_pq,
    create_exact_reranker,
    rerank_ivf_pq_candidates_exact_l2,
)
from timing_utils import measure_synchronized_wall_time

def drop_timing_result(summary):
    result = summary.pop("result", None)
    return result


def search_ivf_pq_with_optional_rerank(
    index,
    queries,
    search_params,
    resources,
    rerank_dataset,
    dataset_ids,
    reranker=None,
):
    search_k = IVFPQ_RERANK_CANDIDATE_K if IVFPQ_ENABLE_EXACT_RERANK else K
    distances, neighbors = search_ivf_pq(
        index,
        queries,
        search_params,
        search_k,
        resources=resources,
    )

    if not IVFPQ_ENABLE_EXACT_RERANK:
        return distances, neighbors

    return rerank_ivf_pq_candidates_exact_l2(
        dataset=rerank_dataset,
        dataset_ids=dataset_ids,
        queries=queries,
        candidate_neighbors=neighbors,
        final_k=K,
        batch_size=IVFPQ_RERANK_BATCH_SIZE,
        reranker=reranker,
    )


def write_results_csv(results, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not results:
        return

    fieldnames = [
        "n_lists",
        "pq_bits",
        "pq_dim",
        "n_probes",
        "build_time",
        "search_time",
        "queries_per_second",
        "latency_per_query",
        "online_median_latency_ms",
        "online_p95_latency_ms",
        "recall_at_10",
        "total_correct",
        "total_possible",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for result in results:
            writer.writerow({
                "n_lists": result["n_lists"],
                "pq_bits": result["pq_bits"],
                "pq_dim": result["pq_dim"],
                "n_probes": result["n_probes"],
                "build_time": result["build_time"],
                "search_time": result["search_time"],
                "queries_per_second": result["queries_per_second"],
                "latency_per_query": result["latency_per_query"],
                "online_median_latency_ms": result["online_summary"]["median_sec"] * MS_PER_SECOND / ONLINE_QUERY_COUNT,
                "online_p95_latency_ms": result["online_summary"]["p95_sec"] * MS_PER_SECOND / ONLINE_QUERY_COUNT,
                "recall_at_10": result["recall_at_10"],
                "total_correct": result["total_correct"],
                "total_possible": result["total_possible"],
            })

def run_ivf_pq_sweep_benchmark():
    """Run the IVF-PQ sweep, save a CSV summary, and return collected metrics."""
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

    rerank_dataset = np.asarray(dataset, dtype=np.float32)

    dataset = dataset.astype(dtype_from_config(IVFPQ_DATASET_DTYPE), copy=False)
    queries = queries.astype(dtype_from_config(IVFPQ_QUERY_DTYPE), copy=False)

    resources = create_multi_gpu_resources()
    sync_fn = resources.sync

    benchmark_queries = queries[:OFFLINE_QUERY_COUNT]
    online_queries = queries[:ONLINE_QUERY_COUNT]
    num_queries = benchmark_queries.shape[0]

    print("\nRunning benchmark on IVF-PQ with multi-GPU search...")
    print("Dataset shape:", dataset.shape)
    print("IVF-PQ dataset dtype:", dataset.dtype)
    print("Queries shape:", benchmark_queries.shape)
    print("IVF-PQ queries dtype:", benchmark_queries.dtype)
    print("k:", K)
    print("n_lists sweep:", IVFPQ_N_LISTS_SWEEP)
    print("pq_bits sweep:", IVFPQ_PQ_BITS_SWEEP)
    print("pq_dim sweep:", IVFPQ_PQ_DIM_SWEEP)
    print("n_probes sweep:", IVFPQ_N_PROBES_SWEEP)

    results = []
    total_build_configs = (len(IVFPQ_N_LISTS_SWEEP) * len(IVFPQ_PQ_BITS_SWEEP) * len(IVFPQ_PQ_DIM_SWEEP))
    total_configs = total_build_configs * len(IVFPQ_N_PROBES_SWEEP)
    config_number = 0

    for n_lists in IVFPQ_N_LISTS_SWEEP:
        for pq_bits in IVFPQ_PQ_BITS_SWEEP:
            for pq_dim in IVFPQ_PQ_DIM_SWEEP:
                index_params = create_index_params(
                    n_lists=n_lists,
                    pq_bits=pq_bits,
                    pq_dim=pq_dim
                )

                print(
                    f"\nBuilding IVF-PQ index with "
                    f"n_lists={n_lists}, pq_bits={pq_bits}, pq_dim={pq_dim}..."
                )

                index, build_time = build_ivf_pq_index(
                    dataset,
                    index_params,
                    resources=resources,
                    sync_fn=sync_fn,
                    print_info=False,
                )
                print(f"IVF-PQ index built in {build_time:.2f} seconds")

                reranker = None
                if IVFPQ_ENABLE_EXACT_RERANK and IVFPQ_RERANK_BACKEND == "multi_gpu":
                    print("Creating multi-GPU exact reranker...")
                    reranker = create_exact_reranker(
                        dataset=rerank_dataset,
                        dataset_ids=dataset_ids,
                        final_k=K,
                        candidate_k=IVFPQ_RERANK_CANDIDATE_K,
                        batch_size=IVFPQ_RERANK_BATCH_SIZE,
                    )
                    print(f"Exact reranker mode: {getattr(reranker, 'mode', 'unknown')}")

                try:
                    for n_probes in IVFPQ_N_PROBES_SWEEP:
                        config_number += 1
                        search_params = create_search_params(n_probes)

                        print(
                            f"\n[{config_number}/{total_configs}] "
                            f"Searching with n_lists={n_lists}, "
                            f"pq_bits={pq_bits}, pq_dim={pq_dim}, "
                            f"n_probes={n_probes}..."
                        )

                        throughput_range = nvtx.start_range(
                            "ivfpq_offline_throughput",
                            color="green",
                        )
                        try:
                            offline_summary = measure_synchronized_wall_time(
                                lambda: search_ivf_pq_with_optional_rerank(
                                    index,
                                    benchmark_queries,
                                    search_params,
                                    resources,
                                    rerank_dataset,
                                    dataset_ids,
                                    reranker,
                                ),
                                warmup_runs=SEARCH_WARMUP_RUNS,
                                timed_runs=SEARCH_TIMED_RUNS,
                                sync_fn=sync_fn,
                            )
                        finally:
                            nvtx.end_range(throughput_range)

                        distances, neighbors = drop_timing_result(offline_summary)
                        search_time = offline_summary["median_sec"]

                        print("Distances shape:", distances.shape)
                        print("Neighbors shape:", neighbors.shape)
                        print("First query top 10 neighbors:", neighbors[0, :DISPLAY_TOP_K])
                        print("First query top 10 distances:", distances[0, :DISPLAY_TOP_K])

                        online_summary = measure_synchronized_wall_time(
                            lambda: search_ivf_pq_with_optional_rerank(
                                index,
                                online_queries,
                                search_params,
                                resources,
                                rerank_dataset,
                                dataset_ids,
                                reranker,
                            ),
                            warmup_runs=SEARCH_WARMUP_RUNS,
                            timed_runs=SEARCH_TIMED_RUNS,
                            sync_fn=sync_fn,
                        )
                        drop_timing_result(online_summary)

                        queries_per_second = num_queries / search_time
                        latency_per_query = search_time * MS_PER_SECOND / num_queries

                        if IVFPQ_ENABLE_EXACT_RERANK:
                            print("Exact rerank enabled; timed search includes candidate search and rerank.")
                        print(f"Offline batch search median: {offline_summary['median_sec']:.4f} seconds")
                        print(f"Offline batch search mean: {offline_summary['mean_sec']:.4f} seconds")
                        print(f"Offline throughput median: {queries_per_second:.2f} queries/second")
                        print(f"Offline latency per query median: {latency_per_query:.4f} ms")
                        print(
                            f"Online single-query median latency: "
                            f"{online_summary['median_sec'] * MS_PER_SECOND / ONLINE_QUERY_COUNT:.4f} ms"
                        )
                        print(
                            f"Online single-query p95 latency: "
                            f"{online_summary['p95_sec'] * MS_PER_SECOND / ONLINE_QUERY_COUNT:.4f} ms"
                        )

                        recall_at_10, total_correct, total_possible = calculate_recall_at_k(
                            neighbors,
                            gt_neighbors[:num_queries],
                            k=K,
                        )
                        print(
                            f"Recall@10: {recall_at_10:.4f} "
                            f"({total_correct} out of {total_possible} correct neighbors)"
                        )

                        results.append({
                            "n_lists": n_lists,
                            "pq_bits": pq_bits,
                            "pq_dim": pq_dim,
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
                        })

                        del distances, neighbors
                        gc.collect()
                        sync_fn()

                finally:
                    sync_fn()
                    del reranker
                    del index
                    gc.collect()
                    sync_fn()

    output_path = RESULTS_DIR / "ivfpq_sweep_results.csv"
    write_results_csv(results, output_path)
    print(f"\nSaved IVF-PQ sweep results to: {output_path}")

    return results


if __name__ == "__main__":
    run_ivf_pq_sweep_benchmark()
