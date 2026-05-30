import time

import numpy as np
from cuvs.common import MultiGpuResources
from cuvs.neighbors.mg import ivf_pq

from config import (
    DISTRIBUTION_MODE,
    IVFPQ_COARSE_SEARCH_DTYPE,
    IVFPQ_CODEBOOK_KIND,
    IVFPQ_CODES_LAYOUT,
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
from support.timing import sync_all_cuda_devices


def create_multi_gpu_resources(device_ids=None):
    return MultiGpuResources(device_ids=device_ids)


def dtype_from_config(name):
    if name == "float32":
        return np.float32
    if name == "float16":
        return np.float16
    raise ValueError(f"Unsupported dtype config value: {name}")


def create_index_params(n_lists, pq_bits, pq_dim):
    return ivf_pq.IndexParams(
        distribution_mode=DISTRIBUTION_MODE,
        n_lists=n_lists,
        metric=METRIC,
        kmeans_n_iters=IVFPQ_KMEANS_N_ITERS,
        kmeans_trainset_fraction=IVFPQ_KMEANS_TRAINSET_FRACTION,
        pq_bits=pq_bits,
        pq_dim=pq_dim,
        codebook_kind=IVFPQ_CODEBOOK_KIND,
        force_random_rotation=IVFPQ_FORCE_RANDOM_ROTATION,
        conservative_memory_allocation=IVFPQ_CONSERVATIVE_MEMORY_ALLOCATION,
        max_train_points_per_pq_code=IVFPQ_MAX_TRAIN_POINTS_PER_PQ_CODE,
        codes_layout=IVFPQ_CODES_LAYOUT,
    )


def build_ivf_pq_index(dataset, index_params, resources=None, sync_fn=None):
    if sync_fn is None:
        sync_fn = resources.sync if resources is not None else sync_all_cuda_devices

    dataset = np.asarray(dataset, dtype=dtype_from_config(IVFPQ_DATASET_DTYPE))

    sync_fn()
    build_start = time.perf_counter()
    index = ivf_pq.build(index_params, dataset, resources=resources)
    sync_fn()

    return index, time.perf_counter() - build_start


def create_search_params(n_probes):
    return ivf_pq.SearchParams(
        n_probes=n_probes,
        search_mode=SEARCH_MODE,
        merge_mode=MERGE_MODE,
        n_rows_per_batch=1000,
        lut_dtype=dtype_from_config(IVFPQ_LUT_DTYPE),
        internal_distance_dtype=dtype_from_config(IVFPQ_INTERNAL_DISTANCE_DTYPE),
        coarse_search_dtype=dtype_from_config(IVFPQ_COARSE_SEARCH_DTYPE),
        max_internal_batch_size=IVFPQ_MAX_INTERNAL_BATCH_SIZE,
    )


def search_ivf_pq(index, queries, search_params, k, resources=None):
    queries = np.asarray(queries, dtype=dtype_from_config(IVFPQ_QUERY_DTYPE))
    return ivf_pq.search(search_params, index, queries, k, resources=resources)
