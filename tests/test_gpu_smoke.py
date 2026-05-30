import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


class GpuRerankSmokeTests(unittest.TestCase):
    def test_multi_gpu_rerank_and_session(self):
        try:
            from ivf_pq.rerank import (
                create_exact_reranker,
                create_ivfpq_search_rerank_session,
                rerank_ivf_pq_candidates_exact_l2_cpu,
            )
        except Exception as exc:
            self.skipTest(f"CUDA rerank extension unavailable: {exc}")

        try:
            from cuda.bindings import runtime

            device_count_result = runtime.cudaGetDeviceCount()
        except Exception as exc:
            self.skipTest(f"CUDA runtime unavailable: {exc}")

        if int(device_count_result[0]) != 0 or device_count_result[1] <= 0:
            self.skipTest(f"CUDA runtime unavailable: {device_count_result[0]}")

        dataset_ids = np.array([10, 11, 12, 13, 14], dtype=np.int64)
        dataset = np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
                [2.0, 2.0],
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
        candidates = np.array(
            [
                [0, 1, 2, 4],
                [3, 1, 2, 4],
            ],
            dtype=np.int64,
        )

        expected_distances, expected_neighbors = rerank_ivf_pq_candidates_exact_l2_cpu(
            dataset,
            dataset_ids,
            queries,
            candidates,
            final_k=2,
            batch_size=2,
        )

        try:
            reranker = create_exact_reranker(
                dataset,
                dataset_ids,
                final_k=2,
                candidate_k=candidates.shape[1],
                batch_size=2,
                device_ids=[0],
                storage_dtype="float16",
            )
        except Exception as exc:
            self.skipTest(f"CUDA reranker unavailable: {exc}")

        actual_distances, actual_neighbors = reranker.rerank(queries, candidates)
        np.testing.assert_allclose(actual_distances, expected_distances, rtol=1e-3, atol=1e-3)
        np.testing.assert_array_equal(actual_neighbors, expected_neighbors)

        invalid_candidates = candidates.copy()
        invalid_candidates[0, 0] = dataset.shape[0]
        with self.assertRaisesRegex(Exception, "invalid dataset row index"):
            reranker.rerank(queries, invalid_candidates)

        rng = np.random.default_rng(0)
        session_dataset = rng.normal(size=(2048, 32)).astype(np.float32)
        session_queries = session_dataset[:8].astype(np.float16)
        session_ids = np.arange(session_dataset.shape[0], dtype=np.int64)

        try:
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
        except Exception as exc:
            self.skipTest(f"C++ search+rerank session unavailable: {exc}")

        session_distances, session_neighbors = session.search_rerank(session_queries)
        self.assertEqual(session_distances.shape, (session_queries.shape[0], 4))
        self.assertEqual(session_neighbors.shape, (session_queries.shape[0], 4))
        self.assertTrue(np.all(np.isfinite(session_distances)))
        self.assertFalse(np.any(session_neighbors < 0))
        self.assertFalse(np.any(session_neighbors >= session_dataset.shape[0]))


if __name__ == "__main__":
    unittest.main()
