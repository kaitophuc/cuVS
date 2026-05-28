# Profiling Notes

These notes collect the commands used while profiling the multi-GPU IVF-PQ benchmark.

Relevant cuVS multi-GPU documentation:
https://docs.rapids.ai/api/cuvs/stable/c_api/neighbors_mg/#multi-gpu-cagra

## Nsight Systems

Run from the repository root:

```bash
nsys profile \
  --trace=cuda,nvtx,osrt,cublas,cusparse \
  --cuda-memory-usage=true \
  --gpu-metrics-devices=all \
  --sample=cpu \
  --cpuctxsw=process-tree \
  --stats=true \
  --force-overwrite=true \
  -o profiles/nsys_ivfpq_2048_8_1536_32 \
  python src/run_ivf_pq_optimized.py
```

## Current Analysis Focus

The main profiling question is how much time is spent in:

- IVF-PQ candidate search on the GPUs.
- Exact squared-L2 reranking and rerank data movement.
- Data movement between host memory and GPU memory.
- Result merging across GPUs.

The current optimized entrypoint is:

```bash
python src/run_ivf_pq_optimized.py
```

## Rebuild CUDA Rerank Extension

The optimized exact rerank backend is built from `src/rerank/cuda/ivfpq_gpu_rerank.cu`.
The compiled extension is written to `src/rerank/extensions/`.

```bash
/home/phuc/Work/cuVS/.conda/bin/python src/rerank/build_extension.py
```

The reranker uses resident GPU dataset shards when memory allows. On smaller GPUs it
automatically falls back to staged mode, which keeps reusable CUDA buffers and pinned
host staging while packing only the candidates owned by each GPU.

Run the optimized multi-GPU rerank benchmark with:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
CUVS_BENCH_IVFPQ_RERANK_BACKEND=multi_gpu \
/home/phuc/Work/cuVS/.conda/bin/python src/run_ivf_pq_optimized.py
```
