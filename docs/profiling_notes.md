# Profiling Notes

These notes collect the profiling commands used for the optimized IVF-PQ benchmark.

## Main Command

```bash
python src/run_ivf_pq_optimized.py
```

The optimized settings live in `src/config.py` under `IVFPQ_OPTIMIZED_PARAMS`.

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
  -o profiles/nsys_ivfpq_optimized \
  python src/run_ivf_pq_optimized.py
```

Useful things to inspect:

- IVF-PQ candidate search time.
- Exact rerank kernel time.
- Host/device copies around rerank.
- Multi-GPU result merging.

## Rebuild CUDA Rerank Extension

```bash
python src/ivf_pq/rerank/build_extension.py
```

The extension writes to `src/ivf_pq/rerank/extensions/`. The default benchmark uses the
C++ session backend, which runs cuVS IVF-PQ search and resident float16 exact rerank in
one extension path.

## Device Selection

Use the same multi-GPU backend for one or more devices:

```bash
CUVS_BENCH_IVFPQ_RERANK_DEVICE_IDS=0,1 \
python src/run_ivf_pq_optimized.py
```
