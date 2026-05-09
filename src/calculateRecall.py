from config import DISPLAY_TOP_K, GROUND_TRUTH_BATCH_SIZE, GROUND_TRUTH_TOP_K, K, QUERY_LIMIT

def calculate_recall_at_k(ivf_neighbors, gt_neighbors, k = K):
    num_queries = gt_neighbors.shape[0]

    ivf_top_k = ivf_neighbors[:num_queries, :k]
    gt_top_k = gt_neighbors[:, :k]

    total_correct = 0

    for query_idx in range(num_queries):
        ivf_ids = set(ivf_top_k[query_idx])
        gt_ids = set(gt_top_k[query_idx])

        correct_for_query = len(ivf_ids.intersection(gt_ids))
        total_correct += correct_for_query

    total_possible = num_queries * k
    recall_at_k = total_correct / total_possible

    return recall_at_k, total_correct, total_possible

def main():
    from IVF_flat import N_PROBES_SWEEP, describe_ivf_params
    from computeL2 import compute_exact_ground_truth
    from load_data import load_default_data
    from multiGPU_IVF_FLat import (
        build_ivf_flat_index,
        create_index_params,
        create_multi_gpu_resources,
        create_search_params,
        search_ivf_flat,
    )

    resources = create_multi_gpu_resources()
    dataset_ids, dataset, _, queries = load_default_data(print_info=True)
    describe_ivf_params(dataset, print_info=True)

    index_params = create_index_params()
    index, _ = build_ivf_flat_index(dataset, index_params, resources=resources, sync_fn=resources.sync, print_info=True)

    search_params = create_search_params(N_PROBES_SWEEP[-1])
    _, neighbors = search_ivf_flat(index, queries, search_params, K, resources=resources, print_info=True)

    gt_distances, gt_neighbors = compute_exact_ground_truth(
        dataset=dataset,
        dataset_ids=dataset_ids,
        queries=queries,
        top_k=GROUND_TRUTH_TOP_K,
        query_limit=QUERY_LIMIT,
        batch_size=GROUND_TRUTH_BATCH_SIZE,
    )

    print("Ground truth distances shape:", gt_distances.shape)
    print("Ground truth neighbors shape:", gt_neighbors.shape)
    print("First query top 10 neighbors:", gt_neighbors[0, :DISPLAY_TOP_K])
    print("First query top 10 distances:", gt_distances[0, :DISPLAY_TOP_K])

    recall_at_10, total_correct, total_possible = calculate_recall_at_k(neighbors, gt_neighbors, k=K)

    print(f"\nRecall@10: {recall_at_10:.4f} ({total_correct} out of {total_possible} correct neighbors)")

if __name__ == "__main__":
    main()
