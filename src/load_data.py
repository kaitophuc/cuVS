import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from config import (
    DATA_CACHE_DIR,
    DATA_CACHE_MMAP,
    EMBEDDING_COLUMN,
    ID_COLUMN,
    PROJECT_ROOT,
    QUERY_PATH,
    TRAIN_PATHS,
    USE_DATA_CACHE,
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
        "resolved_path": str(resolved_path),
    }

    if resolved_path.exists():
        stat = resolved_path.stat()
        fingerprint.update({
            "size_bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        })

    return fingerprint


def data_cache_metadata():
    return {
        "train_paths": [_path_fingerprint(path) for path in TRAIN_PATHS],
        "query_path": _path_fingerprint(QUERY_PATH),
        "vector_dim": VECTOR_DIM,
        "id_column": ID_COLUMN,
        "embedding_column": EMBEDDING_COLUMN,
    }


def data_cache_path():
    metadata = data_cache_metadata()
    metadata_json = json.dumps(metadata, sort_keys=True)
    cache_key = hashlib.sha256(metadata_json.encode("utf-8")).hexdigest()[:16]
    cache_dir = _resolve_config_path(DATA_CACHE_DIR) / f"default_data_{cache_key}"

    return cache_dir, metadata

def load_parquet_vectors(path):
    table = pq.read_table(str(path), columns=[ID_COLUMN, EMBEDDING_COLUMN])

    ids = np.asarray(table[ID_COLUMN].to_numpy(), dtype=np.int64)

    num_rows = table.num_rows

    emb_col = table[EMBEDDING_COLUMN].combine_chunks()
    values = emb_col.values.to_numpy(zero_copy_only=False)
    embeddings = values.reshape(num_rows, VECTOR_DIM).astype(np.float32)

    return ids, embeddings

def load_parquet_vector_files(paths):
    ids_list = []
    embeddings_list = []

    for path in paths:
        ids, embeddings = load_parquet_vectors(path)
        ids_list.append(ids)
        embeddings_list.append(embeddings)

    all_ids = np.concatenate(ids_list, axis=0)
    all_embeddings = np.concatenate(embeddings_list, axis=0)

    return all_ids, all_embeddings


def load_default_data_from_parquet(print_info=False):
    dataset_ids, dataset = load_parquet_vector_files(TRAIN_PATHS)
    query_ids, queries = load_parquet_vectors(QUERY_PATH)

    if print_info:
        print_loaded_data_summary(dataset_ids, dataset, query_ids, queries)

    return dataset_ids, dataset, query_ids, queries


def save_data_cache(cache_dir, dataset_ids, dataset, query_ids, queries, metadata):
    cache_dir.mkdir(parents=True, exist_ok=True)

    np.save(cache_dir / "dataset_ids.npy", dataset_ids)
    np.save(cache_dir / "dataset.npy", dataset)
    np.save(cache_dir / "query_ids.npy", query_ids)
    np.save(cache_dir / "queries.npy", queries)

    with (cache_dir / "metadata.json").open("w", encoding="utf-8") as metadata_file:
        json.dump(metadata, metadata_file, sort_keys=True, indent=2)


def load_data_cache(cache_dir, mmap=DATA_CACHE_MMAP):
    mmap_mode = "r" if mmap else None
    metadata_path = cache_dir / "metadata.json"

    with metadata_path.open("r", encoding="utf-8") as metadata_file:
        metadata = json.load(metadata_file)

    dataset_ids = np.load(cache_dir / "dataset_ids.npy", mmap_mode=mmap_mode)
    dataset = np.load(cache_dir / "dataset.npy", mmap_mode=mmap_mode)
    query_ids = np.load(cache_dir / "query_ids.npy", mmap_mode=mmap_mode)
    queries = np.load(cache_dir / "queries.npy", mmap_mode=mmap_mode)

    return dataset_ids, dataset, query_ids, queries, metadata


def get_or_load_cached_data(
    print_info=False,
    force_recompute=False,
    use_cache=USE_DATA_CACHE,
    mmap=DATA_CACHE_MMAP,
):
    if not use_cache:
        if print_info:
            print("Data cache disabled. Loading data from Parquet.")
        return load_default_data_from_parquet(print_info=print_info)

    cache_dir, metadata = data_cache_path()

    if cache_dir.exists() and not force_recompute:
        dataset_ids, dataset, query_ids, queries, cached_metadata = load_data_cache(
            cache_dir,
            mmap=mmap,
        )
        if cached_metadata == metadata:
            if print_info:
                print(f"Found existing data cache. Loaded: {cache_dir}")
                print_loaded_data_summary(dataset_ids, dataset, query_ids, queries)
            return dataset_ids, dataset, query_ids, queries

        if print_info:
            print(f"Ignoring stale data cache: {cache_dir}")

    if print_info:
        print(f"Building data cache from Parquet: {cache_dir}")

    dataset_ids, dataset, query_ids, queries = load_default_data_from_parquet(
        print_info=print_info,
    )
    save_data_cache(cache_dir, dataset_ids, dataset, query_ids, queries, metadata)

    if print_info:
        print(f"Saved data cache: {cache_dir}")

    if mmap:
        dataset_ids, dataset, query_ids, queries, _ = load_data_cache(cache_dir, mmap=mmap)

    return dataset_ids, dataset, query_ids, queries


def load_default_data(print_info=False):
    return get_or_load_cached_data(print_info=print_info)


def print_loaded_data_summary(dataset_ids, dataset, query_ids, queries):
    print("Dataset IDs shape:", dataset_ids.shape)
    print("Dataset shape:", dataset.shape)
    print("Dataset dtype:", dataset.dtype)

    print("Query IDs shape:", query_ids.shape)
    print("Queries shape:", queries.shape)
    print("Queries dtype:", queries.dtype)

    print("First dataset id:", dataset_ids[0])
    print("Last dataset id:", dataset_ids[-1])
    print("First query id:", query_ids[0])
    print("Embedding dimension:", dataset.shape[1])

def main():
    load_default_data(print_info=True)

if __name__ == "__main__":
    main()
