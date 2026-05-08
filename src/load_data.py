import numpy as np
import pyarrow.parquet as pq
from config import EMBEDDING_COLUMN, ID_COLUMN, QUERY_PATH, TRAIN_PATH, VECTOR_DIM

def load_parquet_vectors(path):
    table = pq.read_table(path, columns=[ID_COLUMN, EMBEDDING_COLUMN])

    ids = np.asarray(table[ID_COLUMN].to_numpy(), dtype=np.int64)

    num_rows = table.num_rows

    emb_col = table[EMBEDDING_COLUMN].combine_chunks()
    values = emb_col.values.to_numpy(zero_copy_only=False)
    embeddings = values.reshape(num_rows, VECTOR_DIM).astype(np.float32)

    return ids, embeddings

def load_default_data(print_info=False):
    dataset_ids, dataset = load_parquet_vectors(TRAIN_PATH)
    query_ids, queries = load_parquet_vectors(QUERY_PATH)

    if print_info:
        print_loaded_data_summary(dataset_ids, dataset, query_ids, queries)

    return dataset_ids, dataset, query_ids, queries

def print_loaded_data_summary(dataset_ids, dataset, query_ids, queries):
    print("Dataset IDs shape:", dataset_ids.shape)
    print("Dataset shape:", dataset.shape)
    print("Dataset dtype:", dataset.dtype)

    print("Query IDs shape:", query_ids.shape)
    print("Queries shape:", queries.shape)
    print("Queries dtype:", queries.dtype)

    print("First dataset id:", dataset_ids[0])
    print("First query id:", query_ids[0])
    print("Embedding dimension:", dataset.shape[1])

def main():
    load_default_data(print_info=True)

if __name__ == "__main__":
    main()
