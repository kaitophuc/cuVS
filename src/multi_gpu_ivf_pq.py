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
    IVFPQ_RERANK_BACKEND,
    MERGE_MODE,
    METRIC,
    SEARCH_MODE,
    IVFPQ_RERANK_BATCH_SIZE,
    IVFPQ_RERANK_DEVICE_ID,
    IVFPQ_RERANK_DEVICE_IDS,
)
from timing_utils import sync_all_cuda_devices
from cuvs.common import MultiGpuResources
from cuvs.neighbors.mg import ivf_pq

_gpu_rerank_fn = None
_multi_gpu_reranker_cls = None

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

def _load_gpu_rerank_fn():
    global _gpu_rerank_fn

    if _gpu_rerank_fn is None:
        from ivfpq_gpu_rerank import rerank_ivf_pq_candidates_exact_l2_gpu

        _gpu_rerank_fn = rerank_ivf_pq_candidates_exact_l2_gpu

    return _gpu_rerank_fn


def _load_multi_gpu_reranker_cls():
    global _multi_gpu_reranker_cls

    if _multi_gpu_reranker_cls is None:
        from ivfpq_gpu_rerank import MultiGpuExactReranker

        _multi_gpu_reranker_cls = MultiGpuExactReranker

    return _multi_gpu_reranker_cls


def create_exact_reranker(
    dataset,
    dataset_ids,
    final_k,
    candidate_k,
    batch_size=IVFPQ_RERANK_BATCH_SIZE,
    device_ids=IVFPQ_RERANK_DEVICE_IDS,
):
    """Create a stateful CUDA exact reranker that uploads dataset shards once."""
    reranker_cls = _load_multi_gpu_reranker_cls()
    return reranker_cls(
        np.asarray(dataset, dtype=np.float32),
        np.asarray(dataset_ids, dtype=np.int64),
        final_k,
        candidate_k,
        batch_size,
        device_ids,
    )


def rerank_ivf_pq_candidates_exact_l2_gpu(
    dataset,
    dataset_ids,
    queries,
    candidate_neighbors,
    final_k,
    batch_size=IVFPQ_RERANK_BATCH_SIZE,
    device_id=IVFPQ_RERANK_DEVICE_ID,
):
    """Rerank IVF-PQ candidate ids with exact squared L2 using one CUDA device."""
    gpu_rerank_fn = _load_gpu_rerank_fn()
    return gpu_rerank_fn(
        np.asarray(dataset, dtype=np.float32),
        np.asarray(dataset_ids, dtype=np.int64),
        np.asarray(queries, dtype=np.float32),
        np.asarray(candidate_neighbors, dtype=np.int64),
        final_k,
        batch_size,
        device_id,
    )


def rerank_ivf_pq_candidates_exact_l2_multi_gpu(
    reranker,
    queries,
    candidate_neighbors,
):
    """Rerank candidates with a preloaded multi-GPU exact reranker."""
    if reranker is None:
        raise ValueError("multi_gpu rerank backend requires a pre-created reranker")

    return reranker.rerank(
        np.asarray(queries, dtype=np.float32),
        np.asarray(candidate_neighbors, dtype=np.int64),
    )


def rerank_ivf_pq_candidates_exact_l2_cpu(
    dataset,
    dataset_ids,
    queries,
    candidate_neighbors,
    final_k,
    batch_size=IVFPQ_RERANK_BATCH_SIZE,
):
    """Rerank IVF-PQ candidate ids with exact host-side squared L2 distance."""
    dataset = np.asarray(dataset, dtype=np.float32)
    dataset_ids = np.asarray(dataset_ids)
    queries = np.asarray(queries, dtype=np.float32)
    candidate_neighbors = np.asarray(candidate_neighbors)

    num_queries, candidate_k = candidate_neighbors.shape

    if final_k > candidate_k:
        raise ValueError(
            f"final_k={final_k} cannot be larger than candidate_k={candidate_k}"
        )
    
    reranked_distances = np.empty((num_queries, final_k), dtype=np.float32)
    reranked_neighbors = np.empty((num_queries, final_k), dtype=dataset_ids.dtype)

    for start in range(0, num_queries, batch_size):
        end = min(start + batch_size, num_queries)

        query_batch = queries[start:end]
        candidate_rows = candidate_neighbors[start:end].astype(np.int64, copy=False)

        candidate_vectors = dataset[candidate_rows]

        query_norms = np.sum(query_batch * query_batch, axis = 1)[:, None]
        candidate_norms = np.sum(candidate_vectors * candidate_vectors, axis = 2)
        dot_products = np.einsum(
            "bcd,bd->bc",
            candidate_vectors,
            query_batch,
            optimize=True,
        )

        exact_distances = query_norms + candidate_norms - 2.0 * dot_products
        exact_distances = np.maximum(exact_distances, 0.0).astype(np.float32)

        top_positions = np.argpartition(
            exact_distances,
            final_k - 1,
            axis=1,
        )[:, :final_k]

        top_distances = np.take_along_axis(
            exact_distances,
            top_positions,
            axis=1,
        )

        sorted_order = np.argsort(top_distances, axis=1)

        sorted_positions = np.take_along_axis(
            top_positions,
            sorted_order,
            axis=1,
        )

        sorted_distances = np.take_along_axis(
            top_distances,
            sorted_order,
            axis=1
        )

        sorted_rows = np.take_along_axis(
            candidate_rows,
            sorted_positions,
            axis=1,
        )

        reranked_distances[start:end] = sorted_distances
        reranked_neighbors[start:end] = dataset_ids[sorted_rows]

    return reranked_distances, reranked_neighbors


def rerank_ivf_pq_candidates_exact_l2(
    dataset,
    dataset_ids,
    queries,
    candidate_neighbors,
    final_k,
    batch_size=IVFPQ_RERANK_BATCH_SIZE,
    backend=IVFPQ_RERANK_BACKEND,
    reranker=None,
):
    """Rerank IVF-PQ candidate ids using the configured exact rerank backend."""
    if backend == "multi_gpu":
        if reranker is None:
            reranker = create_exact_reranker(
                dataset=dataset,
                dataset_ids=dataset_ids,
                final_k=final_k,
                candidate_k=np.asarray(candidate_neighbors).shape[1],
                batch_size=batch_size,
            )
        return rerank_ivf_pq_candidates_exact_l2_multi_gpu(
            reranker=reranker,
            queries=queries,
            candidate_neighbors=candidate_neighbors,
        )

    if backend == "gpu":
        return rerank_ivf_pq_candidates_exact_l2_gpu(
            dataset=dataset,
            dataset_ids=dataset_ids,
            queries=queries,
            candidate_neighbors=candidate_neighbors,
            final_k=final_k,
            batch_size=batch_size,
        )

    if backend == "cpu":
        return rerank_ivf_pq_candidates_exact_l2_cpu(
            dataset=dataset,
            dataset_ids=dataset_ids,
            queries=queries,
            candidate_neighbors=candidate_neighbors,
            final_k=final_k,
            batch_size=batch_size,
        )

    raise ValueError(
        f"Unsupported IVFPQ_RERANK_BACKEND={backend!r}; expected 'multi_gpu', 'gpu', or 'cpu'"
    )
