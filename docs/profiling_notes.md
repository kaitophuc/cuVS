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
- Host-side exact squared-L2 reranking.
- Data movement between host memory and GPU memory.
- Result merging across GPUs.

The current optimized entrypoint is:

```bash
python src/run_ivf_pq_optimized.py
```
