import numpy as np
from cuvs.neighbors.mg import ivf_flat
from config import (
    DISTRIBUTION_MODE,
    IVFPQ_EXAMPLE_DIM,
    IVFPQ_EXAMPLE_N_PROBES,
    IVFPQ_EXAMPLE_NUM_QUERIES,
    IVFPQ_EXAMPLE_NUM_SAMPLES,
    K,
    MERGE_MODE,
    METRIC,
    N_LISTS,
    SEARCH_MODE,
)

n_samples = IVFPQ_EXAMPLE_NUM_SAMPLES
dim = IVFPQ_EXAMPLE_DIM
k = K

dataset = np.random.random((n_samples, dim)).astype(np.float32)
queries = np.random.random((IVFPQ_EXAMPLE_NUM_QUERIES, dim)).astype(np.float32)

index_params = ivf_flat.IndexParams(
    distribution_mode=DISTRIBUTION_MODE,
    n_lists=N_LISTS,
    metric=METRIC
)

index = ivf_flat.build(index_params, dataset)

search_params = ivf_flat.SearchParams(
    n_probes=IVFPQ_EXAMPLE_N_PROBES,
    search_mode=SEARCH_MODE,
    merge_mode=MERGE_MODE,
)

distances, neighbors = ivf_flat.search(search_params, index, queries, k)
print("Distances shape:", distances.shape)
print("Neighbors shape:", neighbors.shape)
