import numpy as np
from cuvs.neighbors.mg import ivf_flat

n_samples = 100000
dim = 96
k = 10

dataset = np.random.random((n_samples, dim)).astype(np.float32)
queries = np.random.random((10000, dim)).astype(np.float32)

index_params = ivf_flat.IndexParams(
    distribution_mode="sharded",
    n_lists=1024,
    metric="sqeuclidean"
)

index = ivf_flat.build(index_params, dataset)

search_params = ivf_flat.SearchParams(
    n_probes=20,
    search_mode="load_balancer",
    merge_mode="merge_on_root_rank",
)

distances, neighbors = ivf_flat.search(search_params, index, queries, k)
print("Distances shape:", distances.shape)
print("Neighbors shape:", neighbors.shape)