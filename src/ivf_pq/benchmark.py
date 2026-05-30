import csv
import gc
import time
from collections import defaultdict

import numpy as np
import nvtx

from config import (
    DISPLAY_TOP_K,
    GROUND_TRUTH_BATCH_SIZE,
    GROUND_TRUTH_TOP_K,
    IVFPQ_DATASET_DTYPE,
    IVFPQ_ENABLE_EXACT_RERANK,
    IVFPQ_QUERY_DTYPE,
    IVFPQ_RERANK_BACKEND,
    IVFPQ_RERANK_BATCH_SIZE,
    IVFPQ_RERANK_CANDIDATE_K,
    IVFPQ_RERANK_DEVICE_IDS,
    IVFPQ_RERANK_STORAGE_DTYPE,
    MS_PER_SECOND,
    OFFLINE_QUERY_COUNT,
    ONLINE_QUERY_COUNT,
    QUERY_LIMIT,
    SEARCH_TIMED_RUNS,
    SEARCH_WARMUP_RUNS,
    K,
)
from ivf_pq.index import (
    build_ivf_pq_index,
    create_index_params,
    create_multi_gpu_resources,
    create_search_params,
    dtype_from_config,
    search_ivf_pq,
)
from ivf_pq.rerank import (
    create_exact_reranker,
    create_ivfpq_search_rerank_session,
    rerank_ivf_pq_candidates_exact_l2,
)
from support.data import load_default_data
from support.ground_truth import get_or_compute_exact_ground_truth
from support.metrics import calculate_recall_at_k
from support.timing import measure_synchronized_wall_time, sync_all_cuda_devices


def run_ivf_pq_configs(configs, output_path):
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

    rerank_dataset = np.asarray(dataset, dtype=np.float32)
    index_dataset = dataset.astype(dtype_from_config(IVFPQ_DATASET_DTYPE), copy=False)
    search_queries = queries.astype(dtype_from_config(IVFPQ_QUERY_DTYPE), copy=False)

    use_session_backend = IVFPQ_ENABLE_EXACT_RERANK and IVFPQ_RERANK_BACKEND == "session"
    resources = None if use_session_backend else create_multi_gpu_resources(IVFPQ_RERANK_DEVICE_IDS)
    sync_fn = sync_all_cuda_devices if use_session_backend else resources.sync

    benchmark_queries = search_queries[:OFFLINE_QUERY_COUNT]
    online_queries = search_queries[:ONLINE_QUERY_COUNT]
    num_queries = benchmark_queries.shape[0]

    print("\nRunning IVF-PQ benchmark")
    print("Dataset shape:", index_dataset.shape)
    print("Dataset dtype:", index_dataset.dtype)
    print("Queries shape:", benchmark_queries.shape)
    print("Queries dtype:", benchmark_queries.dtype)
    print("k:", K)
    print("exact rerank:", IVFPQ_ENABLE_EXACT_RERANK)
    print("rerank backend:", IVFPQ_RERANK_BACKEND)
    print("rerank storage dtype:", IVFPQ_RERANK_STORAGE_DTYPE)

    results = []
    grouped_configs = _group_by_index_params(configs)
    total_configs = len(configs)
    config_number = 0

    for build_params, n_probes_values in grouped_configs.items():
        n_lists, pq_bits, pq_dim = build_params
        index = None
        session = None
        reranker = None

        print(
            f"\nBuilding IVF-PQ index with n_lists={n_lists}, "
            f"pq_bits={pq_bits}, pq_dim={pq_dim}..."
        )

        if use_session_backend:
            sync_fn()
            build_start = time.perf_counter()
            session = create_ivfpq_search_rerank_session(
                index_dataset=index_dataset,
                rerank_dataset=rerank_dataset,
                dataset_ids=dataset_ids,
                final_k=K,
                candidate_k=IVFPQ_RERANK_CANDIDATE_K,
                batch_size=IVFPQ_RERANK_BATCH_SIZE,
                device_ids=IVFPQ_RERANK_DEVICE_IDS,
                n_lists=n_lists,
                pq_bits=pq_bits,
                pq_dim=pq_dim,
                n_probes=n_probes_values[0],
                storage_dtype=IVFPQ_RERANK_STORAGE_DTYPE,
            )
            sync_fn()
            build_time = time.perf_counter() - build_start
            print(f"Session mode: {getattr(session, 'mode', 'unknown')}")
        else:
            index_params = create_index_params(
                n_lists=n_lists,
                pq_bits=pq_bits,
                pq_dim=pq_dim,
            )
            index, build_time = build_ivf_pq_index(
                index_dataset,
                index_params,
                resources=resources,
                sync_fn=sync_fn,
            )

        print(f"IVF-PQ index built in {build_time:.2f} seconds")

        if (
            not use_session_backend
            and IVFPQ_ENABLE_EXACT_RERANK
            and IVFPQ_RERANK_BACKEND == "multi_gpu"
        ):
            print(f"Creating exact reranker (storage={IVFPQ_RERANK_STORAGE_DTYPE})...")
            reranker = create_exact_reranker(
                dataset=rerank_dataset,
                dataset_ids=dataset_ids,
                final_k=K,
                candidate_k=IVFPQ_RERANK_CANDIDATE_K,
                batch_size=IVFPQ_RERANK_BATCH_SIZE,
                device_ids=IVFPQ_RERANK_DEVICE_IDS,
                storage_dtype=IVFPQ_RERANK_STORAGE_DTYPE,
            )
            print(f"Exact reranker mode: {getattr(reranker, 'mode', 'unknown')}")

        try:
            for n_probes in n_probes_values:
                config_number += 1
                print(f"\n[{config_number}/{total_configs}] n_probes={n_probes}")

                if use_session_backend:
                    session.set_n_probes(n_probes)
                    offline_search, online_search = _session_search_fns(
                        session,
                        benchmark_queries,
                        online_queries,
                    )
                else:
                    search_params = create_search_params(n_probes)
                    offline_search, online_search = _python_search_fns(
                        index,
                        search_params,
                        resources,
                        rerank_dataset,
                        dataset_ids,
                        reranker,
                        benchmark_queries,
                        online_queries,
                    )

                throughput_range = nvtx.start_range("ivfpq_offline_throughput", color="green")
                try:
                    offline_summary = measure_synchronized_wall_time(
                        offline_search,
                        warmup_runs=SEARCH_WARMUP_RUNS,
                        timed_runs=SEARCH_TIMED_RUNS,
                        sync_fn=sync_fn,
                    )
                finally:
                    nvtx.end_range(throughput_range)

                distances, neighbors = _take_timing_result(offline_summary)
                search_time = offline_summary["median_sec"]

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
                print("First query top distances:", distances[0, :DISPLAY_TOP_K])
                print(f"Offline median: {search_time:.4f} seconds")
                print(f"Throughput: {queries_per_second:.2f} queries/second")
                print(f"Latency per query: {latency_per_query:.4f} ms")
                print(f"Recall@10: {recall_at_10:.4f} ({total_correct}/{total_possible})")

                results.append(
                    {
                        "n_lists": n_lists,
                        "pq_bits": pq_bits,
                        "pq_dim": pq_dim,
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
            del session, reranker, index
            gc.collect()
            sync_fn()

    write_results_csv(results, output_path)
    print(f"\nSaved IVF-PQ results to: {output_path}")
    return results


def search_ivf_pq_with_rerank(
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
        backend=IVFPQ_RERANK_BACKEND,
        reranker=reranker,
    )


def write_results_csv(results, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
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
            writer.writerow(
                {
                    "n_lists": result["n_lists"],
                    "pq_bits": result["pq_bits"],
                    "pq_dim": result["pq_dim"],
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


def _group_by_index_params(configs):
    grouped = defaultdict(list)
    for config in configs:
        key = (config["n_lists"], config["pq_bits"], config["pq_dim"])
        grouped[key].append(config["n_probes"])
    return grouped


def _session_search_fns(session, benchmark_queries, online_queries):
    def offline_search():
        return session.search_rerank(benchmark_queries)

    def online_search():
        return session.search_rerank(online_queries)

    return offline_search, online_search


def _python_search_fns(
    index,
    search_params,
    resources,
    rerank_dataset,
    dataset_ids,
    reranker,
    benchmark_queries,
    online_queries,
):
    def offline_search():
        return search_ivf_pq_with_rerank(
            index,
            benchmark_queries,
            search_params,
            resources,
            rerank_dataset,
            dataset_ids,
            reranker,
        )

    def online_search():
        return search_ivf_pq_with_rerank(
            index,
            online_queries,
            search_params,
            resources,
            rerank_dataset,
            dataset_ids,
            reranker,
        )

    return offline_search, online_search


def _online_ms(result, field):
    return result["online_summary"][field] * MS_PER_SECOND / ONLINE_QUERY_COUNT


def _take_timing_result(summary):
    return summary.pop("result")
