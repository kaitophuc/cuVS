import faiss
import numpy as np
from config import (
    CAGRA_DIM,
    CAGRA_GRAPH_DEGREE,
    CAGRA_HNSW_M,
    CAGRA_INTERMEDIATE_GRAPH_DEGREE,
    CAGRA_NEW_VECTORS,
    CAGRA_NUM_QUERIES,
    CAGRA_NUM_VECTORS,
    K,
)

#Step 1: Create the CAGRA index config
config = faiss.GpuIndexCagraConfig()
config.graph_degree = CAGRA_GRAPH_DEGREE
config.intermediate_graph_degree = CAGRA_INTERMEDIATE_GRAPH_DEGREE

#Step 2: Create the CAGRA index on the GPU
res = faiss.StandardGpuResources()
gpu_cagra_index = faiss.GpuIndexCagra(res, CAGRA_DIM, faiss.METRIC_L2, config)

#Step 3: Add the 1M vectors to the index
n = CAGRA_NUM_VECTORS
data = np.random.random((n, CAGRA_DIM)).astype('float32')
gpu_cagra_index.add(data)

#Step 4: Search the index for top 10 neighbors of each query
xq = np.random.random((CAGRA_NUM_QUERIES, CAGRA_DIM)).astype('float32')
D, I = gpu_cagra_index.search(xq, K)

#Create the HNSW index object for vectors with 96 dimensions
M = CAGRA_HNSW_M
cpu_hnsw_index = faiss.IndexHNSWCagra(CAGRA_DIM, M, faiss.METRIC_L2)
cpu_hnsw_index.base_level_only = False

#Initialize the HNSW base layer with the same graph degree as the CAGRA index
gpu_cagra_index.copy_to(cpu_hnsw_index)

#Add new vectors to the hierarchy
newVecs = np.random.random((CAGRA_NEW_VECTORS, CAGRA_DIM)).astype('float32')
cpu_hnsw_index.add(newVecs)
