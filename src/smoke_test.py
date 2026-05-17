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


if __name__ == "__main__":
    main()
