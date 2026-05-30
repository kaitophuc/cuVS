import csv
import gc

from config import (
    DISPLAY_TOP_K,
    GROUND_TRUTH_BATCH_SIZE,
    GROUND_TRUTH_TOP_K,
    IVF_FLAT_N_LISTS_SWEEP,
    IVF_FLAT_N_PROBES_SWEEP,
    IVF_FLAT_RESULTS_CSV,
    MS_PER_SECOND,
    OFFLINE_QUERY_COUNT,
    ONLINE_QUERY_COUNT,
    QUERY_LIMIT,
    SEARCH_TIMED_RUNS,
    SEARCH_WARMUP_RUNS,
    K,
)
from ivf_flat.index import (
    build_ivf_flat_index,
    create_index_params,
    create_multi_gpu_resources,
    create_search_params,
    search_ivf_flat,
)
from support.charts import plot_ivf_flat_results
from support.data import load_default_data
from support.ground_truth import get_or_compute_exact_ground_truth
from support.metrics import calculate_recall_at_k
from support.timing import measure_synchronized_wall_time


def run_sweep(output_path=IVF_FLAT_RESULTS_CSV, draw_chart=True):
    dataset_ids, dataset, _, queries = load_default_data(print_info=True)
    _, ground_truth_neighbors = get_or_compute_exact_ground_truth(
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

    print("\nRunning IVF-Flat sweep")
    print("Dataset shape:", dataset.shape)
    print("Queries shape:", benchmark_queries.shape)
    print("n_lists sweep:", IVF_FLAT_N_LISTS_SWEEP)
    print("n_probes sweep:", IVF_FLAT_N_PROBES_SWEEP)

    results = []
    config_number = 0
    total_configs = len(IVF_FLAT_N_LISTS_SWEEP) * len(IVF_FLAT_N_PROBES_SWEEP)

    for n_lists in IVF_FLAT_N_LISTS_SWEEP:
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
            for n_probes in IVF_FLAT_N_PROBES_SWEEP:
                config_number += 1
                search_params = create_search_params(n_probes)
                print(f"\n[{config_number}/{total_configs}] n_probes={n_probes}")

                def offline_search(index=index, search_params=search_params):
                    return search_ivf_flat(
                        index,
                        benchmark_queries,
                        search_params,
                        K,
                        resources=resources,
                    )

                offline_summary = measure_synchronized_wall_time(
                    offline_search,
                    warmup_runs=SEARCH_WARMUP_RUNS,
                    timed_runs=SEARCH_TIMED_RUNS,
                    sync_fn=sync_fn,
                )
                distances, neighbors = _take_timing_result(offline_summary)
                search_time = offline_summary["median_sec"]

                def online_search(index=index, search_params=search_params):
                    return search_ivf_flat(
                        index,
                        online_queries,
                        search_params,
                        K,
                        resources=resources,
                    )

                online_summary = measure_synchronized_wall_time(
                    online_search,
                    warmup_runs=SEARCH_WARMUP_RUNS,
                    timed_runs=SEARCH_TIMED_RUNS,
                    sync_fn=sync_fn,
                )
                _take_timing_result(online_summary)

                queries_per_second = num_queries / search_time
                latency_per_query = search_time * MS_PER_SECOND / num_queries
                recall_at_10, total_correct, total_possible = calculate_recall_at_k(
                    neighbors,
                    ground_truth_neighbors[:num_queries],
                    k=K,
                )

                print("First query top neighbors:", neighbors[0, :DISPLAY_TOP_K])
                print(f"Offline median: {search_time:.4f} seconds")
                print(f"Throughput: {queries_per_second:.2f} queries/second")
                print(f"Recall@10: {recall_at_10:.4f} ({total_correct}/{total_possible})")

                results.append(
                    {
                        "n_lists": n_lists,
                        "n_probes": n_probes,
                        "build_time": build_time,
                        "search_time": search_time,
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

    write_results_csv(results, output_path)
    print(f"\nSaved IVF-Flat sweep results to: {output_path}")

    if draw_chart:
        plot_ivf_flat_results(results)

    return results


def write_results_csv(results, output_path=IVF_FLAT_RESULTS_CSV):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "n_lists",
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
            writer.writerow(
                {
                    "n_lists": result["n_lists"],
                    "n_probes": result["n_probes"],
                    "build_time": result["build_time"],
                    "search_time": result["search_time"],
                    "queries_per_second": result["queries_per_second"],
                    "latency_per_query": result["latency_per_query"],
                    "online_median_latency_ms": _online_ms(result, "median_sec"),
                    "online_p95_latency_ms": _online_ms(result, "p95_sec"),
                    "recall_at_10": result["recall_at_10"],
                    "total_correct": result["total_correct"],
                    "total_possible": result["total_possible"],
                }
            )


def _online_ms(result, field):
    return result["online_summary"][field] * MS_PER_SECOND / ONLINE_QUERY_COUNT


def _take_timing_result(summary):
    return summary.pop("result")
