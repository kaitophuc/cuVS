import csv
import gc
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
import nvtx
from cuvs.common import MultiGpuResources
from cuvs.neighbors import cagra as single_cagra
from cuvs.neighbors.mg import cagra

from config import (
    DISPLAY_TOP_K,
    DISTRIBUTION_MODE,
    GROUND_TRUTH_BATCH_SIZE,
    GROUND_TRUTH_TOP_K,
    K,
    MERGE_MODE,
    METRIC,
    MS_PER_SECOND,
    OFFLINE_QUERY_COUNT,
    ONLINE_QUERY_COUNT,
    QUERY_LIMIT,
    RESULTS_DIR,
    SEARCH_MODE,
    SEARCH_TIMED_RUNS,
    SEARCH_WARMUP_RUNS,
) 
from support.data import load_default_data
from support.ground_truth import get_or_compute_exact_ground_truth
from support.metrics import calculate_recall_at_k
from support.timing import measure_synchronized_wall_time
from ivf_pq.rerank import create_exact_reranker, rerank_ivf_pq_candidates_exact_l2

def _optional_int_list_from_env(name):
    raw_value = os.environ.get(name, "")
    if not raw_value.strip():
        return None
    return [int(value.strip()) for value in raw_value.split(",") if value.strip()]

CAGRA_RESULTS_CSV = RESULTS_DIR / "cagra_optimized_results.csv"

CAGRA_DATASET_DTYPE = os.environ.get("CUVS_BENCH_CAGRA_DATASET_DTYPE", "float16")
CAGRA_QUERY_DTYPE = os.environ.get("CUVS_BENCH_CAGRA_QUERY_DTYPE", CAGRA_DATASET_DTYPE)
CAGRA_DEVICE_IDS = _optional_int_list_from_env("CUVS_BENCH_CAGRA_DEVICE_IDS")
MERGE_MODE = "tree_merge"
CAGRA_ENABLE_EXACT_RERANK = os.environ.get("CUVS_BENCH_CAGRA_ENABLE_EXACT_RERANK", "1") != "0"
CAGRA_RERANK_CANDIDATE_K = int(os.environ.get("CUVS_BENCH_CAGRA_RERANK_CANDIDATE_K", "48"))
CAGRA_RERANK_BATCH_SIZE = int(os.environ.get("CUVS_BENCH_CAGRA_RERANK_BATCH_SIZE", "512"))
CAGRA_RERANK_BACKEND = os.environ.get("CUVS_BENCH_CAGRA_RERANK_BACKEND", "multi_gpu")
CAGRA_RERANK_DEVICE_IDS = (
    _optional_int_list_from_env("CUVS_BENCH_CAGRA_RERANK_DEVICE_IDS")
    or CAGRA_DEVICE_IDS
)
CAGRA_RERANK_STORAGE_DTYPE = os.environ.get("CUVS_BENCH_CAGRA_RERANK_STORAGE_DTYPE", "float16")

CAGRA_OPTIMIZED_PARAMS = {
    "graph_degree": 32,
    "intermediate_graph_degree": 64,
    "build_algo": "ivf_pq",

    "compression_enabled": True,
    "compression_pq_bits": 8,
    "compression_pq_dim": 384,

    "enable_exact_rerank": True,
    "rerank_candidate_k": 48,

    "itopk_size": 48,
    "max_queries": 256,
    "max_iterations": 0,
    "algo": "auto",
    "team_size": 0,
    "search_width": 1,
    "min_iterations": 0,
    "thread_block_size": 0,
    "hashmap_mode": "auto",
    "hashmap_min_bitlen": 0,
    "hashmap_max_fill_rate": 0.5,
    "num_random_samplings": 1,
    "rand_xor_mask": 0x128394,
    "n_rows_per_batch": 256,
}

def dtype_from_config(name):
    if name == "float32":
        return np.float32
    if name == "float16":
        return np.float16
    if name == "int8":
        return np.int8
    if name == "uint8":
        return np.uint8
    raise ValueError(f"Unsupported CAGRA dtype: {name}")

def as_contiguous_dtype(array, dtype_name):
    return np.ascontiguousarray(np.asarray(array, dtype=dtype_from_config(dtype_name)))

def create_compression_params(config):
    if not config.get("compression_enabled", False):
        return None
    return single_cagra.CompressionParams(
        pq_bits=config["compression_pq_bits"],
        pq_dim=config["compression_pq_dim"],
    )

def create_index_params(config):
    if METRIC not in {"sqeuclidean", "inner_product"}:
        raise ValueError(
            "Multi-GPU CAGRA supports sqeuclidean and inner_product here; "
            f"config.METRIC is {METRIC!r}"
        )
    
    compression = create_compression_params(config)

    return cagra.IndexParams(
        distribution_mode=DISTRIBUTION_MODE,
        metric=METRIC,
        graph_degree=config["graph_degree"],
        intermediate_graph_degree=config["intermediate_graph_degree"],
        build_algo=config["build_algo"],
        compression=compression,
    )

def create_search_params(config):
    search_k = get_search_k(config)
    if config["itopk_size"] < search_k:
        raise ValueError(
            "CAGRA itopk_size should be at least the requested search_k "
            f"({config['itopk_size']} < {search_k})"
        )

    return cagra.SearchParams(
        search_mode=SEARCH_MODE,
        merge_mode=MERGE_MODE,
        n_rows_per_batch=config["n_rows_per_batch"],
        itopk_size=config["itopk_size"],
        max_queries=config["max_queries"],
        max_iterations=config["max_iterations"],
        algo=config["algo"],
        team_size=config["team_size"],
        search_width=config["search_width"],
        min_iterations=config["min_iterations"],
        thread_block_size=config["thread_block_size"],
        hashmap_mode=config["hashmap_mode"],
        hashmap_min_bitlen=config["hashmap_min_bitlen"],
        hashmap_max_fill_rate=config["hashmap_max_fill_rate"],
        num_random_samplings=config["num_random_samplings"],
        rand_xor_mask=config["rand_xor_mask"],
    )

def exact_rerank_enabled(config):
    return config.get("enable_exact_rerank", CAGRA_ENABLE_EXACT_RERANK)

def get_search_k(config):
    if not exact_rerank_enabled(config):
        return K
    return max(K, config.get("rerank_candidate_k", CAGRA_RERANK_CANDIDATE_K))

def validate_rerank_backend():
    if CAGRA_RERANK_BACKEND not in {"multi_gpu", "cpu"}:
        raise ValueError(
            "CUVS_BENCH_CAGRA_RERANK_BACKEND must be 'multi_gpu' or 'cpu'. "
            "The IVF-PQ 'session' backend owns IVF-PQ search and cannot rerank "
            "standalone CAGRA candidates."
        )

def create_cagra_exact_reranker(dataset, dataset_ids, candidate_k):
    try:
        return (
            create_exact_reranker(
                dataset=dataset,
                dataset_ids=dataset_ids,
                final_k=K,
                candidate_k=candidate_k,
                batch_size=CAGRA_RERANK_BATCH_SIZE,
                device_ids=CAGRA_RERANK_DEVICE_IDS,
                storage_dtype=CAGRA_RERANK_STORAGE_DTYPE,
            ),
            CAGRA_RERANK_STORAGE_DTYPE,
        )
    except RuntimeError as exc:
        if CAGRA_RERANK_STORAGE_DTYPE == "float32" or "does not fit" not in str(exc):
            raise

        print(
            "Resident float16 reranker did not fit beside the CAGRA index; "
            "retrying with float32 auto/staged storage."
        )
        return (
            create_exact_reranker(
                dataset=dataset,
                dataset_ids=dataset_ids,
                final_k=K,
                candidate_k=candidate_k,
                batch_size=CAGRA_RERANK_BATCH_SIZE,
                device_ids=CAGRA_RERANK_DEVICE_IDS,
                storage_dtype="float32",
            ),
            "float32",
        )

def build_cagra_index(dataset, index_params, resources, sync_fn):
    sync_fn()
    build_start = time.perf_counter()
    index = cagra.build(index_params, dataset, resources=resources)
    sync_fn()

    if not getattr(index, "trained", True):
        raise RuntimeError("CAGRA index was not trained")
    
    return index, time.perf_counter() - build_start

def copy_dataset_ids_for_row_neighbors(row_neighbors, dataset_ids, out):
    if row_neighbors.size:
        min_row = int(row_neighbors.min())
        max_row = int(row_neighbors.max())
        if min_row < 0 or max_row >= dataset_ids.shape[0]:
            raise ValueError(
                f"CAGRA returned row neighbor outside dataset range: "
                f"min={min_row}, max={max_row}, rows={dataset_ids.shape[0]}"
            )
        
    out[...] = dataset_ids[row_neighbors]
    return out

def make_cagra_search_fn(
    index,
    queries,
    rerank_queries,
    search_params,
    final_k,
    search_k,
    resources,
    sync_fn,
    dataset_ids,
    rerank_dataset,
    exact_rerank,
    reranker=None,
):
    row_neighbors = np.empty((queries.shape[0], search_k), dtype=np.int64)
    candidate_distances = np.empty((queries.shape[0], search_k), dtype=np.float32)
    id_neighbors = np.empty((queries.shape[0], final_k), dtype=np.asarray(dataset_ids).dtype)
    distances = np.empty((queries.shape[0], final_k), dtype=np.float32)

    def run_search():
        cagra.search(
            search_params,
            index,
            queries,
            search_k,
            neighbors=row_neighbors,
            distances=candidate_distances,
            resources=resources
        )

        sync_fn()
        if exact_rerank:
            return rerank_ivf_pq_candidates_exact_l2(
                dataset=rerank_dataset,
                dataset_ids=dataset_ids,
                queries=rerank_queries,
                candidate_neighbors=row_neighbors,
                final_k=final_k,
                batch_size=CAGRA_RERANK_BATCH_SIZE,
                backend=CAGRA_RERANK_BACKEND,
                reranker=reranker,
            )

        distances[...] = candidate_distances[:, :final_k]
        copy_dataset_ids_for_row_neighbors(row_neighbors[:, :final_k], dataset_ids, id_neighbors)
        return distances, id_neighbors

    return run_search

def run_cagra_configs(configs, output_path=CAGRA_RESULTS_CSV):
    resources = MultiGpuResources(device_ids=CAGRA_DEVICE_IDS)
    sync_fn = resources.sync

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

    index_dataset = as_contiguous_dtype(dataset, CAGRA_DATASET_DTYPE)
    search_queries = as_contiguous_dtype(queries, CAGRA_QUERY_DTYPE)
    rerank_dataset = np.asarray(dataset, dtype=np.float32)
    rerank_queries = np.asarray(queries, dtype=np.float32)

    benchmark_queries = search_queries[:OFFLINE_QUERY_COUNT]
    online_queries = search_queries[:ONLINE_QUERY_COUNT]
    benchmark_rerank_queries = rerank_queries[:OFFLINE_QUERY_COUNT]
    online_rerank_queries = rerank_queries[:ONLINE_QUERY_COUNT]
    num_queries = benchmark_queries.shape[0]

    print("\nRunning CAGRA benchmark")
    print("Dataset shape:", index_dataset.shape)
    print("Dataset dtype:", index_dataset.dtype)
    print("Queries shape:", benchmark_queries.shape)
    print("Queries dtype:", benchmark_queries.dtype)
    print("k:", K)
    print("distribution mode:", DISTRIBUTION_MODE)
    print("search mode:", SEARCH_MODE)
    print("merge mode:", MERGE_MODE)
    print("device ids:", CAGRA_DEVICE_IDS if CAGRA_DEVICE_IDS is not None else "all visible")
    print("exact rerank default:", CAGRA_ENABLE_EXACT_RERANK)
    print("rerank backend:", CAGRA_RERANK_BACKEND)
    print("rerank storage dtype:", CAGRA_RERANK_STORAGE_DTYPE)
    validate_rerank_backend()

    results = []
    grouped_configs = _group_by_build_params(configs)
    total_configs = len(configs)
    config_number = 0

    for build_key, search_configs in grouped_configs.items():
        build_config = _build_config_from_key(build_key)
        index = None
        reranker = None
        reranker_candidate_k = None
        reranker_storage_dtype = "none"

        print(
            "\nBuilding CAGRA index with "
            f"graph_degree={build_config['graph_degree']}, "
            f"intermediate_graph_degree={build_config['intermediate_graph_degree']}, "
            f"build_algo={build_config['build_algo']}..."
        )

        index_params = create_index_params(build_config)
        index, build_time = build_cagra_index(
            index_dataset,
            index_params,
            resources=resources,
            sync_fn=sync_fn,
        )
        print(f"CAGRA index built in {build_time:.2f} seconds")

        try:
            for config in search_configs:
                config_number += 1
                print(
                    f"\n[{config_number}/{total_configs}] "
                    f"itopk_size={config['itopk_size']}, "
                    f"search_k={get_search_k(config)}, "
                    f"search_width={config['search_width']}, "
                    f"max_iterations={config['max_iterations']}"
                )

                exact_rerank = exact_rerank_enabled(config)
                search_k = get_search_k(config)
                if exact_rerank and CAGRA_RERANK_BACKEND == "multi_gpu":
                    if reranker_candidate_k != search_k:
                        sync_fn()
                        reranker = None
                        gc.collect()
                        sync_fn()
                        print(
                            "Creating exact CAGRA reranker "
                            f"(candidate_k={search_k}, storage={CAGRA_RERANK_STORAGE_DTYPE})..."
                        )
                        reranker, reranker_storage_dtype = create_cagra_exact_reranker(
                            dataset=rerank_dataset,
                            dataset_ids=dataset_ids,
                            candidate_k=search_k,
                        )
                        reranker_candidate_k = search_k
                        print(
                            f"Exact reranker mode: {getattr(reranker, 'mode', 'unknown')} "
                            f"(storage={reranker_storage_dtype})"
                        )

                search_params = create_search_params(config)
                offline_search = make_cagra_search_fn(
                    index=index,
                    queries=benchmark_queries,
                    rerank_queries=benchmark_rerank_queries,
                    search_params=search_params,
                    final_k=K,
                    search_k=search_k,
                    resources=resources,
                    sync_fn=sync_fn,
                    dataset_ids=dataset_ids,
                    rerank_dataset=rerank_dataset,
                    exact_rerank=exact_rerank,
                    reranker=reranker,
                )
                online_search = make_cagra_search_fn(
                    index=index,
                    queries=online_queries,
                    rerank_queries=online_rerank_queries,
                    search_params=search_params,
                    final_k=K,
                    search_k=search_k,
                    resources=resources,
                    sync_fn=sync_fn,
                    dataset_ids=dataset_ids,
                    rerank_dataset=rerank_dataset,
                    exact_rerank=exact_rerank,
                    reranker=reranker,
                )

                throughput_range = nvtx.start_range("cagra_offline_throughput", color="blue")
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
                        **config,
                        "search_k": search_k,
                        "exact_rerank": exact_rerank,
                        "rerank_backend": CAGRA_RERANK_BACKEND if exact_rerank else "none",
                        "rerank_storage_dtype": (
                            reranker_storage_dtype if exact_rerank else "none"
                        ),
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
            reranker = None
            index = None
            gc.collect()
            sync_fn()

    write_results_csv(results, output_path)
    print(f"\nSaved CAGRA results to: {output_path}")
    return results

def write_results_csv(results, output_path=CAGRA_RESULTS_CSV):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "graph_degree",
        "intermediate_graph_degree",
        "build_algo",
        "compression_enabled",
        "compression_pq_bits",
        "compression_pq_dim",
        "enable_exact_rerank",
        "rerank_candidate_k",
        "search_k",
        "exact_rerank",
        "rerank_backend",
        "rerank_storage_dtype",
        "itopk_size",
        "max_queries",
        "max_iterations",
        "algo",
        "team_size",
        "search_width",
        "min_iterations",
        "thread_block_size",
        "hashmap_mode",
        "hashmap_min_bitlen",
        "hashmap_max_fill_rate",
        "num_random_samplings",
        "rand_xor_mask",
        "n_rows_per_batch",
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
                    **{field: result[field] for field in fieldnames if field in result},
                    "online_median_latency_ms": _online_ms(result, "median_sec"),
                    "online_p95_latency_ms": _online_ms(result, "p95_sec"),
                }
            )

def _group_by_build_params(configs):
    grouped = defaultdict(list)
    for config in configs:
        key = (
            config["graph_degree"],
            config["intermediate_graph_degree"],
            config["build_algo"],
            config.get("compression_enabled", False),
            config.get("compression_pq_bits", 0),
            config.get("compression_pq_dim", 0),
        )
        grouped[key].append(config)
    return grouped


def _build_config_from_key(key):
    (graph_degree, intermediate_graph_degree, build_algo, compression_enabled, compression_pq_bits, compression_pq_dim) = key
    return {
        "graph_degree": graph_degree,
        "intermediate_graph_degree": intermediate_graph_degree,
        "build_algo": build_algo,
        "compression_enabled": compression_enabled,
        "compression_pq_bits": compression_pq_bits,
        "compression_pq_dim": compression_pq_dim,
    }


def _online_ms(result, field):
    return result["online_summary"][field] * MS_PER_SECOND / ONLINE_QUERY_COUNT


def _take_timing_result(summary):
    return summary.pop("result")


def main():
    run_cagra_configs([CAGRA_OPTIMIZED_PARAMS], CAGRA_RESULTS_CSV)


if __name__ == "__main__":
    main()
