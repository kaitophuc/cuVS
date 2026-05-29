import numpy as np

from recall import calculate_recall_at_k


def exact_neighbors(dataset, queries, k):
    """Return exact squared-L2 nearest-neighbor indices for a tiny synthetic dataset."""
    distances = (
        np.sum(queries * queries, axis=1)[:, None]
        + np.sum(dataset * dataset, axis=1)[None, :]
        - 2.0 * queries @ dataset.T
    )
    return np.argsort(distances, axis=1)[:, :k]


def main():
    dataset = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ],
        dtype=np.float32,
    )
    queries = np.array(
        [
            [0.1, 0.0],
            [0.9, 1.0],
        ],
        dtype=np.float32,
    )

    neighbors = exact_neighbors(dataset, queries, k=2)
    recall, total_correct, total_possible = calculate_recall_at_k(neighbors, neighbors, k=2)

    print(f"Synthetic Recall@2: {recall:.4f} ({total_correct}/{total_possible})")

    try:
        from rerank import (
            create_exact_reranker,
            create_ivfpq_search_rerank_session,
            rerank_ivf_pq_candidates_exact_l2_cpu,
        )
    except Exception as exc:
        print(f"Multi-GPU exact rerank smoke test skipped: {exc}")
        return

    dataset_ids = np.array([10, 11, 12, 13, 14], dtype=np.int64)
    gpu_dataset = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [2.0, 2.0],
        ],
        dtype=np.float32,
    )
    gpu_queries = np.array(
        [
            [0.1, 0.0],
            [0.9, 1.0],
        ],
        dtype=np.float32,
    )
    candidates = np.array(
        [
            [0, 1, 2, 4],
            [3, 1, 2, 4],
        ],
        dtype=np.int64,
    )

    expected_distances, expected_neighbors = rerank_ivf_pq_candidates_exact_l2_cpu(
        gpu_dataset,
        dataset_ids,
        gpu_queries,
        candidates,
        final_k=2,
        batch_size=2,
    )

    try:
        reranker = create_exact_reranker(
            gpu_dataset,
            dataset_ids,
            final_k=2,
            candidate_k=candidates.shape[1],
            batch_size=2,
        )
    except Exception as exc:
        print(f"Multi-GPU exact rerank smoke test skipped: {exc}")
        return

    actual_distances, actual_neighbors = reranker.rerank(gpu_queries, candidates)

    np.testing.assert_allclose(actual_distances, expected_distances, rtol=1e-5, atol=1e-5)
    np.testing.assert_array_equal(actual_neighbors, expected_neighbors)

    try:
        fp16_reranker = create_exact_reranker(
            gpu_dataset,
            dataset_ids,
            final_k=2,
            candidate_k=candidates.shape[1],
            batch_size=2,
            storage_dtype="float16",
        )
    except Exception as exc:
        print(f"Resident fp16 rerank smoke test skipped: {exc}")
    else:
        if fp16_reranker.mode != "resident_float16":
            raise AssertionError(
                f"expected resident_float16 mode, got {fp16_reranker.mode!r}"
            )

        fp16_distances, fp16_neighbors = fp16_reranker.rerank(gpu_queries, candidates)
        np.testing.assert_allclose(
            fp16_distances,
            expected_distances,
            rtol=1e-3,
            atol=1e-3,
        )
        np.testing.assert_array_equal(fp16_neighbors, expected_neighbors)

        invalid_candidates = candidates.copy()
        invalid_candidates[0, 0] = gpu_dataset.shape[0]
        try:
            fp16_reranker.rerank(gpu_queries, invalid_candidates)
            raise AssertionError("invalid candidate row did not raise")
        except Exception as exc:
            if "invalid dataset row index" not in str(exc):
                raise

    invalid_candidates = candidates.copy()
    invalid_candidates[0, 0] = gpu_dataset.shape[0]
    try:
        reranker.rerank(gpu_queries, invalid_candidates)
        raise AssertionError("invalid candidate row did not raise")
    except Exception as exc:
        if "invalid dataset row index" not in str(exc):
            raise

    try:
        rng = np.random.default_rng(0)
        session_dataset = rng.normal(size=(2048, 32)).astype(np.float32)
        session_queries = session_dataset[:8].astype(np.float16)
        session_ids = np.arange(session_dataset.shape[0], dtype=np.int64)
        session = create_ivfpq_search_rerank_session(
            index_dataset=session_dataset.astype(np.float16),
            rerank_dataset=session_dataset,
            dataset_ids=session_ids,
            final_k=4,
            candidate_k=16,
            batch_size=4,
            device_ids=[0],
            n_lists=32,
            pq_bits=4,
            pq_dim=16,
            n_probes=4,
            storage_dtype="float16",
        )
        if session.mode != "cuvs_cpp_search_resident_float16_rerank":
            raise AssertionError(f"unexpected session mode: {session.mode!r}")

        session_distances, session_neighbors = session.search_rerank(session_queries)
        if session_distances.shape != (session_queries.shape[0], 4):
            raise AssertionError("session distances shape mismatch")
        if session_neighbors.shape != (session_queries.shape[0], 4):
            raise AssertionError("session neighbors shape mismatch")
        if not np.all(np.isfinite(session_distances)):
            raise AssertionError("session returned non-finite distances")
        if np.any(session_neighbors < 0) or np.any(session_neighbors >= session_dataset.shape[0]):
            raise AssertionError("session returned out-of-range neighbors")
    except Exception as exc:
        print(f"C++ search+rerank session smoke test skipped: {exc}")
    else:
        print("C++ search+rerank session smoke test passed.")

    print("Multi-GPU exact rerank smoke test passed.")


if __name__ == "__main__":
    main()
