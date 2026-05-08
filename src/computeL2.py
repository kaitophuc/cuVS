import numpy as np

def compute_exact_ground_truth(dataset, dataset_ids, queries, top_k = 100, query_limit = 100, batch_size = 10, print_progress=True):
    queries_subset = queries[:query_limit]

    gt_distances_batches = []
    gt_neighbors_batches = []

    dataset_norms = np.sum(dataset * dataset, axis=1)

    for start in range(0, query_limit, batch_size):
        end = min(start + batch_size, query_limit)
        query_batch = queries_subset[start:end]

        query_norms = np.sum(query_batch * query_batch, axis = 1)

        distances = (
            query_norms[:, None] + dataset_norms[None, :] - 2.0 * query_batch @ dataset.T
        )

        candidate_indices = np.argpartition(distances, top_k, axis=1)[:, :top_k]

        candidate_distances = np.take_along_axis(distances, candidate_indices, axis=1)

        sorted_order = np.argsort(candidate_distances, axis=1)

        sorted_indices = np.take_along_axis(candidate_indices, sorted_order, axis=1)
        sorted_distances = np.take_along_axis(candidate_distances, sorted_order, axis=1)

        sorted_neighbors_ids = dataset_ids[sorted_indices]

        gt_distances_batches.append(sorted_distances.astype(np.float32))
        gt_neighbors_batches.append(sorted_neighbors_ids.astype(np.int64))

        if print_progress:
            print(f"computed ground truth for queries {start} to {end - 1}")

    gt_distances = np.vstack(gt_distances_batches)
    gt_neighbors = np.vstack(gt_neighbors_batches)

    return gt_distances, gt_neighbors

def main():
    from load_data import load_default_data

    dataset_ids, dataset, _, queries = load_default_data(print_info=True)

    gt_distances, gt_neighbors = compute_exact_ground_truth(
        dataset=dataset,
        dataset_ids=dataset_ids,
        queries=queries,
        top_k=100,
        query_limit=100,
        batch_size=10,
        print_progress=True,
    )

    print("Ground truth distances shape:", gt_distances.shape)
    print("Ground truth neighbors shape:", gt_neighbors.shape)
    print("First query top 10 neighbors:", gt_neighbors[0, :10])
    print("First query top 10 distances:", gt_distances[0, :10])

if __name__ == "__main__":
    main()
