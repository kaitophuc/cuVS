import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from recall import calculate_recall_at_k


class RecallTests(unittest.TestCase):
    def test_recall_at_k_counts_intersections_per_query(self):
        retrieved = np.array([
            [1, 2, 3],
            [4, 5, 6],
        ])
        ground_truth = np.array([
            [2, 9, 1],
            [7, 5, 6],
        ])

        recall, total_correct, total_possible = calculate_recall_at_k(
            retrieved,
            ground_truth,
            k=2,
        )

        self.assertEqual(total_correct, 2)
        self.assertEqual(total_possible, 4)
        self.assertEqual(recall, 0.5)


if __name__ == "__main__":
    unittest.main()
