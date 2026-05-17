# cuVS Multi-GPU ANN Benchmarks

This repository contains Python benchmarks for approximate nearest-neighbor search with
RAPIDS cuVS multi-GPU indexes. The current focus is comparing IVF-Flat and IVF-PQ on a
5M-vector OpenAI embedding dataset with 1536-dimensional vectors and squared L2 distance.

## Current Baseline

- Dataset: `openai_large_5m`
- Database size: 5,000,000 vectors when all 10 training shards are present
- Query set: 1,000 vectors from `test.parquet`
- Metric: squared L2 / `sqeuclidean`
- Hardware used for the latest local runs: 2 x RTX 5070 Ti, 16 GB each
- Best tracked IVF-PQ run: see `results/ivfpq_sweep_results.csv`

## Repository Layout

```text
src/
  config.py                  Shared benchmark configuration
  load_data.py               Parquet loading and NumPy cache handling
  ground_truth.py            Exact/precomputed ground-truth loading
  recall.py                  Recall@k calculation
  multi_gpu_ivf_flat.py      Multi-GPU IVF-Flat wrappers
  multi_gpu_ivf_pq.py        Multi-GPU IVF-PQ wrappers and exact reranking
  multi_gpu_cagra.py         Multi-GPU CAGRA wrappers
  benchmark.py               Single IVF-Flat benchmark
  sweep_benchmark.py         IVF-Flat parameter sweep
  sweep_ivfpq_benchmark.py   IVF-PQ parameter sweep
  run_ivf_pq_optimized.py    Current best single IVF-PQ run
  smoke_test.py              Tiny NumPy-only sanity check
docs/
  profiling_notes.md         Nsight profiling commands and analysis notes
results/
  ivfpq_sweep_results.csv    Small tracked CSV summary from the latest run
tests/
  test_recall.py             Lightweight unit test for Recall@k
```

Large generated folders are intentionally not tracked: `.conda/`, `openai_large_5m/`,
`data_cache/`, `ground_truth_cache/`, `profiles/`, and `logs/`.

## Environment Setup

Create the conda environment:

```bash
conda env create -f environment.yml
conda activate cuvs-mgpu-benchmark
```

The local environment used for the latest verification had these important versions:

- Python 3.12.13
- cuVS 26.04.00
- RMM 26.04.00
- cuda-python 12.9.6
- NumPy 2.4.3
- PyArrow 24.0.0
- nvtx 0.2.15

## Full Dataset Guide

By default, the benchmark expects the full dataset under:

```text
openai_large_5m/
  train-00-of-10.parquet
  train-01-of-10.parquet
  train-02-of-10.parquet
  train-03-of-10.parquet
  train-04-of-10.parquet
  train-05-of-10.parquet
  train-06-of-10.parquet
  train-07-of-10.parquet
  train-08-of-10.parquet
  train-09-of-10.parquet
  test.parquet
  neighbors.parquet          optional, used as precomputed ground truth
```

Each training and query Parquet file should contain:

- `id`: integer vector id
- `emb`: embedding vector column with dimension 1536

The optional `neighbors.parquet` file should contain:

- `id`: query id
- `neighbors_id`: ordered list of exact nearest-neighbor ids

If the full dataset is too large to keep inside the repository folder, store it elsewhere
and point the benchmark to it:

```bash
export CUVS_BENCH_DATA_DIR=/path/to/openai_large_5m
```

You can also place caches outside the repository:

```bash
export CUVS_BENCH_DATA_CACHE_DIR=/path/to/data_cache
export CUVS_BENCH_GROUND_TRUTH_CACHE_DIR=/path/to/ground_truth_cache
```

The NumPy data cache is enabled by default to avoid repeatedly decoding the large Parquet
files. Disable it only for debugging:

```bash
export CUVS_BENCH_USE_DATA_CACHE=0
```

## Running Benchmarks

Run one IVF-Flat benchmark:

```bash
python src/benchmark.py
```

Run the IVF-Flat sweep:

```bash
python src/sweep_benchmark.py
```

Run the IVF-PQ sweep:

```bash
python src/sweep_ivfpq_benchmark.py
```

Run the current optimized IVF-PQ configuration:

```bash
python src/run_ivf_pq_optimized.py
```

The IVF-PQ sweep writes:

```text
results/ivfpq_sweep_results.csv
```

## Reproducibility Notes

The default benchmark settings live in `src/config.py`. Important values include:

- `K = 10`
- `QUERY_LIMIT = 1000`
- `GROUND_TRUTH_TOP_K = 100`
- `SEARCH_WARMUP_RUNS = 3`
- `SEARCH_TIMED_RUNS = 10`
- `DISTRIBUTION_MODE = "sharded"`
- `SEARCH_MODE = "load_balancer"`
- `MERGE_MODE = "merge_on_root_rank"`

The current IVF-PQ configuration uses float16 inputs/search internals plus optional exact
host-side reranking of `IVFPQ_RERANK_CANDIDATE_K = 100` candidates.

## Tests

Run the tiny NumPy-only smoke test:

```bash
python src/smoke_test.py
```

Run the lightweight unit test:

```bash
python -m unittest discover -s tests
```

This test does not require GPUs or the full dataset. The full benchmark scripts require a
working CUDA/cuVS environment and the dataset files described above.

## Profiling

Nsight Systems commands and profiling notes are in:

```text
docs/profiling_notes.md
```

Generated profiling artifacts should stay in `profiles/`, which is ignored by git.

## License

This project is shared under the MIT License. See `LICENSE`.
