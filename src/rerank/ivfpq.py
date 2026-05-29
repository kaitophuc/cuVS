import numpy as np

from config import (
    IVFPQ_RERANK_BACKEND,
    IVFPQ_RERANK_BATCH_SIZE,
    IVFPQ_RERANK_DEVICE_ID,
    IVFPQ_RERANK_DEVICE_IDS,
    IVFPQ_RERANK_STORAGE_DTYPE,
)


SUPPORTED_RERANK_BACKENDS = {"session", "multi_gpu", "gpu", "cpu"}
SUPPORTED_RERANK_STORAGE_DTYPES = {"float32", "float16"}

_single_gpu_rerank_fn = None
_multi_gpu_reranker_cls = None
_search_rerank_session_cls = None


def _as_float32(array):
    return np.asarray(array, dtype=np.float32)


def _as_int64(array):
    return np.asarray(array, dtype=np.int64)


def _load_single_gpu_rerank_fn():
    global _single_gpu_rerank_fn

    if _single_gpu_rerank_fn is None:
        from rerank.extensions.ivfpq_gpu_rerank import (
            rerank_ivf_pq_candidates_exact_l2_gpu,
        )

        _single_gpu_rerank_fn = rerank_ivf_pq_candidates_exact_l2_gpu

    return _single_gpu_rerank_fn


def _load_multi_gpu_reranker_cls():
    global _multi_gpu_reranker_cls

    if _multi_gpu_reranker_cls is None:
        from rerank.extensions.ivfpq_gpu_rerank import MultiGpuExactReranker

        _multi_gpu_reranker_cls = MultiGpuExactReranker

    return _multi_gpu_reranker_cls


def _load_search_rerank_session_cls():
    global _search_rerank_session_cls

    if _search_rerank_session_cls is None:
        from rerank.extensions.ivfpq_gpu_rerank import IvfPqSearchRerankSession

        _search_rerank_session_cls = IvfPqSearchRerankSession

    return _search_rerank_session_cls


def create_exact_reranker(
    dataset,
    dataset_ids,
    final_k,
    candidate_k,
    batch_size=IVFPQ_RERANK_BATCH_SIZE,
    device_ids=IVFPQ_RERANK_DEVICE_IDS,
    storage_dtype=IVFPQ_RERANK_STORAGE_DTYPE,
):
    """Create a stateful CUDA exact reranker that reuses GPU buffers across calls."""
    if storage_dtype not in SUPPORTED_RERANK_STORAGE_DTYPES:
        expected = ", ".join(sorted(SUPPORTED_RERANK_STORAGE_DTYPES))
        raise ValueError(
            f"Unsupported rerank storage_dtype={storage_dtype!r}; expected one of: {expected}"
        )

    reranker_cls = _load_multi_gpu_reranker_cls()
    return reranker_cls(
        _as_float32(dataset),
        _as_int64(dataset_ids),
        final_k,
        candidate_k,
        batch_size,
        device_ids,
        storage_dtype,
    )


def create_ivfpq_search_rerank_session(
    index_dataset,
    rerank_dataset,
    dataset_ids,
    final_k,
    candidate_k,
    batch_size=IVFPQ_RERANK_BATCH_SIZE,
    device_ids=IVFPQ_RERANK_DEVICE_IDS,
    n_lists=4096,
    pq_bits=4,
    pq_dim=384,
    n_probes=32,
    storage_dtype=IVFPQ_RERANK_STORAGE_DTYPE,
):
    """Create a C++ cuVS IVF-PQ search plus resident fp16 rerank session."""
    if storage_dtype != "float16":
        raise ValueError(
            "The session backend currently supports only "
            "CUVS_BENCH_IVFPQ_RERANK_STORAGE_DTYPE=float16"
        )

    session_cls = _load_search_rerank_session_cls()
    return session_cls(
        np.ascontiguousarray(index_dataset, dtype=np.float16),
        _as_float32(rerank_dataset),
        _as_int64(dataset_ids),
        final_k,
        candidate_k,
        batch_size,
        device_ids,
        n_lists,
        pq_bits,
        pq_dim,
        n_probes,
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
    rerank_fn = _load_single_gpu_rerank_fn()
    return rerank_fn(
        _as_float32(dataset),
        _as_int64(dataset_ids),
        _as_float32(queries),
        _as_int64(candidate_neighbors),
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
        _as_float32(queries),
        _as_int64(candidate_neighbors),
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
    dataset = _as_float32(dataset)
    dataset_ids = np.asarray(dataset_ids)
    queries = _as_float32(queries)
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

        query_norms = np.sum(query_batch * query_batch, axis=1)[:, None]
        candidate_norms = np.sum(candidate_vectors * candidate_vectors, axis=2)
        dot_products = np.einsum(
            "bcd,bd->bc",
            candidate_vectors,
            query_batch,
            optimize=True,
        )

        exact_distances = query_norms + candidate_norms - 2.0 * dot_products
        exact_distances = np.maximum(exact_distances, 0.0).astype(np.float32)

        top_positions = np.argpartition(exact_distances, final_k - 1, axis=1)[:, :final_k]
        top_distances = np.take_along_axis(exact_distances, top_positions, axis=1)
        sorted_order = np.argsort(top_distances, axis=1)

        sorted_positions = np.take_along_axis(top_positions, sorted_order, axis=1)
        sorted_distances = np.take_along_axis(top_distances, sorted_order, axis=1)
        sorted_rows = np.take_along_axis(candidate_rows, sorted_positions, axis=1)

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

    if backend == "session":
        raise ValueError(
            "The session backend owns both IVF-PQ search and rerank; use "
            "create_ivfpq_search_rerank_session(...).search_rerank(queries) "
            "instead of the standalone rerank helper."
        )

    expected = ", ".join(sorted(SUPPORTED_RERANK_BACKENDS))
    raise ValueError(
        f"Unsupported IVFPQ_RERANK_BACKEND={backend!r}; expected one of: {expected}"
    )
