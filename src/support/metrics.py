from config import K


def calculate_recall_at_k(retrieved_neighbors, ground_truth_neighbors, k=K):
    """Compute Recall@k by counting exact id matches for each query."""
    num_queries = ground_truth_neighbors.shape[0]
    retrieved_top_k = retrieved_neighbors[:num_queries, :k]
    ground_truth_top_k = ground_truth_neighbors[:, :k]

    total_correct = 0
    for query_idx in range(num_queries):
        retrieved_ids = set(retrieved_top_k[query_idx])
        ground_truth_ids = set(ground_truth_top_k[query_idx])
        total_correct += len(retrieved_ids.intersection(ground_truth_ids))

    total_possible = num_queries * k
    return total_correct / total_possible, total_correct, total_possible
