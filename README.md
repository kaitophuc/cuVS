database: openai_large_5m/train-00-of-10.parquet
database size: 500000 vectors
dimensions: 1536
queries: openai_large_5m/test.parquet
queries size: 1000 vectors
GPUs: 2 x RTX 5070 Ti, 16 GB each
index: multi-GPU IVF-Flat
metric: squared L2 / sqeuclidean
k: 10