import time

import numpy as np
from config import (
    DISTRIBUTION_MODE,
    IVFPQ_CODEBOOK_KIND,
    IVFPQ_CODES_LAYOUT,
    IVFPQ_COARSE_SEARCH_DTYPE,
    IVFPQ_CONSERVATIVE_MEMORY_ALLOCATION,
    IVFPQ_DATASET_DTYPE,
    IVFPQ_FORCE_RANDOM_ROTATION,
    IVFPQ_INTERNAL_DISTANCE_DTYPE,
    IVFPQ_KMEANS_N_ITERS,
    IVFPQ_KMEANS_TRAINSET_FRACTION,
    IVFPQ_LUT_DTYPE,
    IVFPQ_MAX_INTERNAL_BATCH_SIZE,
    IVFPQ_MAX_TRAIN_POINTS_PER_PQ_CODE,
    IVFPQ_QUERY_DTYPE,
    MERGE_MODE,
    METRIC,
    SEARCH_MODE,
)
# Backward-compatible exports; new code should import rerank helpers from rerank.
from rerank import (
    create_exact_reranker,
    rerank_ivf_pq_candidates_exact_l2,
    rerank_ivf_pq_candidates_exact_l2_cpu,
    rerank_ivf_pq_candidates_exact_l2_gpu,
    rerank_ivf_pq_candidates_exact_l2_multi_gpu,
)
from timing_utils import sync_all_cuda_devices
from cuvs.common import MultiGpuResources
from cuvs.neighbors.mg import ivf_pq

def create_multi_gpu_resources(device_ids=None):
    return MultiGpuResources(device_ids=device_ids)

def dtype_from_config(name):
    if name == "float32":
        return np.float32
    if name == "float16":
        return np.float16
    if name == "uint8":
        return np.uint8
    if name == "int8":
        return np.int8
    
    raise ValueError(f"Unsupported dtype config value: {name}")

def create_index_params(
    distribution_mode=DISTRIBUTION_MODE,
    n_lists=1024,
    metric=METRIC,
    pq_bits=8,
    pq_dim=192,
    kmeans_n_iters=IVFPQ_KMEANS_N_ITERS,
    kmeans_trainset_fraction=IVFPQ_KMEANS_TRAINSET_FRACTION,
    codebook_kind=IVFPQ_CODEBOOK_KIND,
    force_random_rotation=IVFPQ_FORCE_RANDOM_ROTATION,
    conservative_memory_allocation=IVFPQ_CONSERVATIVE_MEMORY_ALLOCATION,
    max_train_points_per_pq_code=IVFPQ_MAX_TRAIN_POINTS_PER_PQ_CODE,
    codes_layout=IVFPQ_CODES_LAYOUT,
):
    return ivf_pq.IndexParams(
        distribution_mode=distribution_mode,
        n_lists=n_lists,
        metric=metric,
        kmeans_n_iters=kmeans_n_iters,
        kmeans_trainset_fraction=kmeans_trainset_fraction,
        pq_bits=pq_bits,
        pq_dim=pq_dim,
        codebook_kind=codebook_kind,
        force_random_rotation=force_random_rotation,
        conservative_memory_allocation=conservative_memory_allocation,
        max_train_points_per_pq_code=max_train_points_per_pq_code,
        codes_layout=codes_layout,
    )

def build_ivf_pq_index(dataset, index_params, resources=None, sync_fn=None, print_info=False):
    """Build a multi-GPU IVF-PQ index and return the index plus elapsed seconds."""
    if print_info:
        print("Building IVF-PQ index...")
        print("distribution_mode:", index_params.distribution_mode)
        print("n_lists:", index_params.n_lists)
        print("metric:", index_params.metric)
        print("pq_bits:", index_params.pq_bits)
        print("pq_dim:", index_params.pq_dim)
        print("codebook_kind:", index_params.codebook_kind)
        print("dataset shape:", dataset.shape)
        print("dataset dtype:", dataset.dtype)
    
    if sync_fn is None:
        sync_fn = resources.sync if resources is not None else sync_all_cuda_devices

    dataset = np.asarray(dataset, dtype=dtype_from_config(IVFPQ_DATASET_DTYPE))

    sync_fn()
    build_start = time.perf_counter()

    index = ivf_pq.build(index_params, dataset, resources=resources)

    sync_fn()
    build_end = time.perf_counter()
    build_time = build_end - build_start

    if print_info:
        print(f"IVF-PQ index built in {build_time:.2f} seconds")

    return index, build_time

def create_search_params(
    n_probes,
    search_mode=SEARCH_MODE,
    merge_mode=MERGE_MODE,
    n_rows_per_batch=1000,
    lut_dtype=IVFPQ_LUT_DTYPE,
    internal_distance_dtype=IVFPQ_INTERNAL_DISTANCE_DTYPE,
    coarse_search_dtype=IVFPQ_COARSE_SEARCH_DTYPE,
    max_internal_batch_size=IVFPQ_MAX_INTERNAL_BATCH_SIZE,
):
    return ivf_pq.SearchParams(
        n_probes=n_probes,
        search_mode=search_mode,
        merge_mode=merge_mode,
        n_rows_per_batch=n_rows_per_batch,
        lut_dtype=dtype_from_config(lut_dtype),
        internal_distance_dtype=dtype_from_config(internal_distance_dtype),
        coarse_search_dtype=dtype_from_config(coarse_search_dtype),
        max_internal_batch_size=max_internal_batch_size,
    )

def search_ivf_pq(index, queries, search_params, k, resources=None, print_info=None):
    """Search a multi-GPU IVF-PQ index and return distances and neighbor ids."""
    if print_info:
        print("\nPerforming IVF-PQ search...")
        print("n_probes:", search_params.n_probes)
        print("search_mode:", search_params.search_mode)
        print("merge_mode:", search_params.merge_mode)
        print("n_rows_per_batch:", search_params.n_rows_per_batch)
        print("max_internal_batch_size:", search_params.max_internal_batch_size)

    queries = np.asarray(queries, dtype=dtype_from_config(IVFPQ_QUERY_DTYPE))

    distances, neighbors = ivf_pq.search(
        search_params,
        index,
        queries,
        k,
        resources=resources,
    )

    if print_info:
        print("Search completed.")
        print("Distances shape:", distances.shape)
        print("Neighbors shape:", neighbors.shape)

    return distances, neighbors
