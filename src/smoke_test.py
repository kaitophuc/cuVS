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

    print("Multi-GPU exact rerank smoke test passed.")


if __name__ == "__main__":
    main()
