# cuVS Multi-GPU ANN Benchmarks

This project benchmarks multi-GPU approximate nearest-neighbor search with RAPIDS cuVS.
The main focus is IVF-PQ and CAGRA on the 5M OpenAI embedding dataset, with IVF-Flat
kept as a comparison baseline for sweeps and report charts.

## Current Baseline

- Dataset: `openai_large_5m`
- Database size: 5,000,000 vectors when all 10 training shards are present
- Query set: 1,000 vectors from `test.parquet`
- Metric: squared L2 / `sqeuclidean`
- Main IVF-PQ run: `python src/run_ivf_pq_optimized.py`
- Main CAGRA Python run: see the CAGRA command in Running Benchmarks
- Main CAGRA C++ run: `./run_cagra_optimized.sh`
- Expected optimized result on the current local setup: about `0.93` Recall@10 and
  roughly `120k` QPS with the fused C++ session backend

## Repository Layout

```text
src/
  config.py                 Shared paths, benchmark settings, and tuned parameters
  run_ivf_pq_optimized.py   Current best single IVF-PQ run
  sweep_ivf_flat.py         IVF-Flat sweep entrypoint
  sweep_ivf_pq.py           IVF-PQ sweep entrypoint
  draw_charts.py            Regenerate charts from existing CSV files
  support/                  Data loading, ground truth, metrics, timing, chart helpers
  cagra/                    CAGRA Python runner, sweep, and standalone C++/CUDA source
  ivf_flat/                 IVF-Flat index helpers and sweep code
  ivf_pq/                   IVF-PQ index helpers, benchmark flow, sweep code
    rerank/                 Exact rerank wrapper and CUDA extension source
tests/                      Recall unit test and CUDA rerank smoke test
docs/                       Profiling notes
results/                    Small tracked result examples and generated charts
```

Large generated folders are intentionally ignored: `.conda/`, `openai_large_5m/`,
`data_cache/`, `ground_truth_cache/`, `profiles/`, and `logs/`.

## Environment Setup

```bash
conda env create -f environment.yml
conda activate cuvs-mgpu-benchmark
```

The local environment used for verification includes Python 3.12, cuVS 26.04, RMM 26.04,
cuda-python 12.9, NumPy 2.4, PyArrow 24, Matplotlib, Ruff, Black, and `nvtx`.

## Dataset

By default, the benchmark expects:

```text
openai_large_5m/
  train-00-of-10.parquet
  ...
  train-09-of-10.parquet
  test.parquet
  neighbors.parquet
```

Training and query Parquet files should have `id` and `emb` columns. The optional
`neighbors.parquet` file should have query `id` and ordered `neighbors_id` columns; when
present, it is used as precomputed ground truth.

Useful environment overrides:

```bash
export CUVS_BENCH_DATA_DIR=/path/to/openai_large_5m
export CUVS_BENCH_DATA_CACHE_DIR=/path/to/data_cache
export CUVS_BENCH_GROUND_TRUTH_CACHE_DIR=/path/to/ground_truth_cache
export CUVS_BENCH_USE_DATA_CACHE=0
```

Most benchmark settings should be changed in `src/config.py` instead of passed through
long command lines.

## Running Benchmarks

Run the optimized IVF-PQ configuration:

```bash
python src/run_ivf_pq_optimized.py
```

This writes:

```text
results/ivfpq_optimized_results.csv
```

Run the optimized CAGRA Python configuration:

```bash
python -c "import sys; from pathlib import Path; sys.path.insert(0, str(Path('src').resolve())); from cagra.run_cagra_optimized import main; main()"
```

This writes:

```text
results/cagra_optimized_results.csv
```

The direct file form `python src/cagra/run_cagra_optimized.py` can trip cuVS
multi-GPU resource initialization on this local RAPIDS/CUDA stack, so the command above
imports the runner as a module instead.

Run the standalone C++/CUDA CAGRA executable:

```bash
./run_cagra_optimized.sh
```

Use `./run_cagra_optimized.sh --build-only` to only rebuild the executable.

Run the sweeps:

```bash
python src/sweep_ivf_flat.py
python src/sweep_ivf_pq.py
python -c "import sys; from pathlib import Path; sys.path.insert(0, str(Path('src').resolve())); from cagra.sweep_cagra import main; main()"
```

The sweep outputs are:

```text
results/ivf_flat_sweep_results.csv
results/ivfpq_sweep_results.csv
results/cagra_sweep_results.csv
results/ivf_flat_qps_vs_recall.png
results/ivfpq_qps_vs_recall.png
```

Regenerate charts from existing CSVs:

```bash
python src/draw_charts.py
```

When both sweep CSVs exist, this also writes:

```text
results/ivf_flat_vs_ivfpq_qps_vs_recall.png
```

## Important Config Values

The optimized IVF-PQ setup is in `IVFPQ_OPTIMIZED_PARAMS`:

```text
n_lists=4096, pq_bits=4, pq_dim=384, n_probes=32
```

The exact rerank path uses `IVFPQ_RERANK_CANDIDATE_K = 100`, `K = 10`, and defaults to:

```text
IVFPQ_RERANK_BACKEND = "session"
IVFPQ_RERANK_STORAGE_DTYPE = "float16"
```

Use `CUVS_BENCH_IVFPQ_RERANK_DEVICE_IDS=0` to run the same backend on one GPU.
Use `CUVS_BENCH_IVFPQ_RERANK_BACKEND=multi_gpu` to run the Python cuVS search plus
standalone CUDA reranker path.
Use `CUVS_BENCH_IVFPQ_RERANK_BACKEND=cpu` only for debugging because it is much slower.

The optimized CAGRA setup lives in `src/cagra/run_cagra_optimized.py`:

```text
graph_degree=32, intermediate_graph_degree=64, build_algo=ivf_pq,
compression_pq_bits=8, compression_pq_dim=384, itopk_size=48
```

CAGRA exact rerank defaults to `CUVS_BENCH_CAGRA_RERANK_BACKEND=multi_gpu`,
`CUVS_BENCH_CAGRA_RERANK_CANDIDATE_K=48`, and
`CUVS_BENCH_CAGRA_RERANK_STORAGE_DTYPE=float16`. Use
`CUVS_BENCH_CAGRA_DEVICE_IDS=0` or `CUVS_BENCH_CAGRA_RERANK_DEVICE_IDS=0` to restrict
the CAGRA index or reranker to one GPU.

## CUDA Rerank Extension

Build the extension after changing the CUDA source:

```bash
python src/ivf_pq/rerank/build_extension.py
```

The compiled module is generated under `src/ivf_pq/rerank/extensions/` and ignored by git.
Rebuild it after CUDA/cuVS environment changes too; a stale extension may fail to import
with an error such as `libcudart.so.12: cannot open shared object file`.

## Tests

```bash
python -m unittest discover -s tests
```

The GPU smoke test skips cleanly when the CUDA/cuVS extension is unavailable.

## Profiling

Nsight Systems commands and notes are in `docs/profiling_notes.md`. Generated profiling
artifacts should stay in `profiles/`.

## License

This project is shared under the MIT License. See `LICENSE`.
