import hashlib
import json
from pathlib import Path

import numpy as np
from config import (
    DISPLAY_TOP_K,
    EMBEDDING_COLUMN,
    GROUND_TRUTH_BATCH_SIZE,
    GROUND_TRUTH_CACHE_DIR,
    GROUND_TRUTH_TOP_K,
    ID_COLUMN,
    METRIC,
    PROJECT_ROOT,
    QUERY_LIMIT,
    QUERY_PATH,
    TRAIN_PATHS,
    VECTOR_DIM,
)


def _resolve_config_path(path):
    raw_path = Path(path)
    if raw_path.is_absolute():
        return raw_path

    return (PROJECT_ROOT / raw_path).resolve()


def _path_fingerprint(path):
    resolved_path = _resolve_config_path(path)
    fingerprint = {
        "configured_path": str(path),
        "resolved_path": str(resolved_path),
    }

    if resolved_path.exists():
        stat = resolved_path.stat()
        fingerprint.update({
            "size_bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        })

    return fingerprint


def ground_truth_cache_metadata(top_k=GROUND_TRUTH_TOP_K, query_limit=QUERY_LIMIT):
    return {
        "train_paths": [_path_fingerprint(path) for path in TRAIN_PATHS],
        "query_path": _path_fingerprint(QUERY_PATH),
        "vector_dim": VECTOR_DIM,
        "id_column": ID_COLUMN,
        "embedding_column": EMBEDDING_COLUMN,
        "metric": METRIC,
        "top_k": top_k,
        "query_limit": query_limit,
    }


def ground_truth_cache_path(top_k=GROUND_TRUTH_TOP_K, query_limit=QUERY_LIMIT):
    metadata = ground_truth_cache_metadata(top_k=top_k, query_limit=query_limit)
    metadata_json = json.dumps(metadata, sort_keys=True)
    cache_key = hashlib.sha256(metadata_json.encode("utf-8")).hexdigest()[:16]

    cache_dir = _resolve_config_path(GROUND_TRUTH_CACHE_DIR)
    file_name = f"ground_truth_q{query_limit}_k{top_k}_{cache_key}.npz"

    return cache_dir / file_name, metadata

def compute_exact_ground_truth(dataset, dataset_ids, queries, top_k = GROUND_TRUTH_TOP_K, query_limit = QUERY_LIMIT, batch_size = GROUND_TRUTH_BATCH_SIZE):
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

    gt_distances = np.vstack(gt_distances_batches)
    gt_neighbors = np.vstack(gt_neighbors_batches)

    return gt_distances, gt_neighbors


def save_ground_truth_cache(path, distances, neighbors, metadata):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        distances=distances,
        neighbors=neighbors,
        metadata_json=json.dumps(metadata, sort_keys=True),
    )


def load_ground_truth_cache(path):
    with np.load(path, allow_pickle=False) as cache:
        distances = cache["distances"]
        neighbors = cache["neighbors"]
        metadata = json.loads(str(cache["metadata_json"]))

    return distances, neighbors, metadata


def get_or_compute_exact_ground_truth(
    dataset,
    dataset_ids,
    queries,
    top_k=GROUND_TRUTH_TOP_K,
    query_limit=QUERY_LIMIT,
    batch_size=GROUND_TRUTH_BATCH_SIZE,
    force_recompute=False,
    print_info=False,
):
    cache_path, metadata = ground_truth_cache_path(top_k=top_k, query_limit=query_limit)

    if cache_path.exists() and not force_recompute:
        distances, neighbors, cached_metadata = load_ground_truth_cache(cache_path)
        if cached_metadata == metadata:
            if print_info:
                print(f"Found existing ground truth cache. Loaded: {cache_path}")
            return distances, neighbors

        if print_info:
            print(f"Ignoring stale ground truth cache: {cache_path}")

    if print_info:
        print(f"Computing exact ground truth and saving cache: {cache_path}")

    distances, neighbors = compute_exact_ground_truth(
        dataset=dataset,
        dataset_ids=dataset_ids,
        queries=queries,
        top_k=top_k,
        query_limit=query_limit,
        batch_size=batch_size,
    )
    save_ground_truth_cache(cache_path, distances, neighbors, metadata)

    return distances, neighbors


def main():
    from load_data import load_default_data

    dataset_ids, dataset, _, queries = load_default_data(print_info=True)

    gt_distances, gt_neighbors = get_or_compute_exact_ground_truth(
        dataset=dataset,
        dataset_ids=dataset_ids,
        queries=queries,
        top_k=GROUND_TRUTH_TOP_K,
        query_limit=QUERY_LIMIT,
        batch_size=GROUND_TRUTH_BATCH_SIZE,
        print_info=True,
    )

    print("Ground truth distances shape:", gt_distances.shape)
    print("Ground truth neighbors shape:", gt_neighbors.shape)
    print("First query top 10 neighbors:", gt_neighbors[0, :DISPLAY_TOP_K])
    print("First query top 10 distances:", gt_distances[0, :DISPLAY_TOP_K])

if __name__ == "__main__":
    main()
