import faiss
import numpy as np

#Step 1: Create the CAGRA index config
config = faiss.GpuIndexCagraConfig()
config.graph_degree = 32
config.intermediate_graph_degree = 64

#Step 2: Create the CAGRA index on the GPU
res = faiss.StandardGpuResources()
gpu_cagra_index = faiss.GpuIndexCagra(res, 96, faiss.METRIC_L2, config)

#Step 3: Add the 1M vectors to the index
n = 1000000
data = np.random.random((n, 96)).astype('float32')
gpu_cagra_index.add(data)

#Step 4: Search the index for top 10 neighbors of each query
xq = np.random.random((10000, 96)).astype('float32')
D, I = gpu_cagra_index.search(xq, 10)

#Create the HNSW index object for vectors with 96 dimensions
M = 16
cpu_hnsw_index = faiss.IndexHNSWCagra(96, M, faiss.METRIC_L2)
cpu_hnsw_index.base_level_only = False

#Initialize the HNSW base layer with the same graph degree as the CAGRA index
gpu_cagra_index.copy_to(cpu_hnsw_index)

#Add new vectors to the hierarchy
newVecs = np.random.random((100000, 96)).astype('float32')
cpu_hnsw_index.add(newVecs)