import os
from pathlib import Path


def _optional_int_list_from_env(name):
    raw_value = os.environ.get(name, "")
    if not raw_value.strip():
        return None
    return [int(value.strip()) for value in raw_value.split(",") if value.strip()]


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(
    os.environ.get("CUVS_BENCH_DATA_DIR", PROJECT_ROOT / "openai_large_5m")
).expanduser()
RESULTS_DIR = PROJECT_ROOT / "results"

TRAIN_PATHS = [DATA_DIR / f"train-{split_id:02d}-of-10.parquet" for split_id in range(10)]
QUERY_PATH = DATA_DIR / "test.parquet"
PRECOMPUTED_NEIGHBORS_PATH = DATA_DIR / "neighbors.parquet"

VECTOR_DIM = 1536
ID_COLUMN = "id"
EMBEDDING_COLUMN = "emb"

DATA_CACHE_DIR = Path(
    os.environ.get("CUVS_BENCH_DATA_CACHE_DIR", PROJECT_ROOT / "data_cache")
).expanduser()
USE_DATA_CACHE = os.environ.get("CUVS_BENCH_USE_DATA_CACHE", "1") != "0"
DATA_CACHE_MMAP = os.environ.get("CUVS_BENCH_DATA_CACHE_MMAP", "1") != "0"
GROUND_TRUTH_CACHE_DIR = Path(
    os.environ.get("CUVS_BENCH_GROUND_TRUTH_CACHE_DIR", PROJECT_ROOT / "ground_truth_cache")
).expanduser()

K = 10
METRIC = "sqeuclidean"
DISTRIBUTION_MODE = "sharded"
SEARCH_MODE = "load_balancer"
MERGE_MODE = "merge_on_root_rank"

GROUND_TRUTH_TOP_K = 100
QUERY_LIMIT = 1000
GROUND_TRUTH_BATCH_SIZE = 10
OFFLINE_QUERY_COUNT = 1000
ONLINE_QUERY_COUNT = 1
SEARCH_WARMUP_RUNS = 3
SEARCH_TIMED_RUNS = 10
DISPLAY_TOP_K = 10
MS_PER_SECOND = 1000.0

IVF_FLAT_N_LISTS_SWEEP = [512, 1024, 2048, 4096]
IVF_FLAT_N_PROBES_SWEEP = [8, 16, 32, 64, 128]
IVF_FLAT_DEFAULT_N_LISTS = IVF_FLAT_N_LISTS_SWEEP[1]

IVFPQ_N_LISTS_SWEEP = [1024, 2048, 4096]
IVFPQ_N_PROBES_SWEEP = [32, 64, 128]
IVFPQ_PQ_BITS_SWEEP = [4, 6, 8]
IVFPQ_PQ_DIM_SWEEP = [384, 768, 1536]

IVFPQ_OPTIMIZED_PARAMS = {
    "n_lists": 4096,
    "pq_bits": 4,
    "pq_dim": 384,
    "n_probes": 32,
}

IVFPQ_KMEANS_N_ITERS = 20
IVFPQ_KMEANS_TRAINSET_FRACTION = 0.5
IVFPQ_CODEBOOK_KIND = "subspace"
IVFPQ_FORCE_RANDOM_ROTATION = False
IVFPQ_CONSERVATIVE_MEMORY_ALLOCATION = True
IVFPQ_MAX_TRAIN_POINTS_PER_PQ_CODE = 256
IVFPQ_CODES_LAYOUT = "interleaved"

IVFPQ_DATASET_DTYPE = "float16"
IVFPQ_QUERY_DTYPE = "float16"
IVFPQ_LUT_DTYPE = "float16"
IVFPQ_INTERNAL_DISTANCE_DTYPE = "float16"
IVFPQ_COARSE_SEARCH_DTYPE = "float16"
IVFPQ_MAX_INTERNAL_BATCH_SIZE = 4096

IVFPQ_ENABLE_EXACT_RERANK = True
IVFPQ_RERANK_CANDIDATE_K = 100
IVFPQ_RERANK_BATCH_SIZE = int(os.environ.get("CUVS_BENCH_IVFPQ_RERANK_BATCH_SIZE", "512"))
IVFPQ_RERANK_BACKEND = os.environ.get("CUVS_BENCH_IVFPQ_RERANK_BACKEND", "session")
IVFPQ_RERANK_DEVICE_IDS = _optional_int_list_from_env("CUVS_BENCH_IVFPQ_RERANK_DEVICE_IDS")
IVFPQ_RERANK_STORAGE_DTYPE = os.environ.get("CUVS_BENCH_IVFPQ_RERANK_STORAGE_DTYPE", "float16")

IVF_FLAT_RESULTS_CSV = RESULTS_DIR / "ivf_flat_sweep_results.csv"
IVFPQ_SWEEP_RESULTS_CSV = RESULTS_DIR / "ivfpq_sweep_results.csv"
IVFPQ_OPTIMIZED_RESULTS_CSV = RESULTS_DIR / "ivfpq_optimized_results.csv"

IVF_FLAT_CHART_PATH = RESULTS_DIR / "ivf_flat_qps_vs_recall.png"
IVFPQ_CHART_PATH = RESULTS_DIR / "ivfpq_qps_vs_recall.png"
IVF_COMPARISON_CHART_PATH = RESULTS_DIR / "ivf_flat_vs_ivfpq_qps_vs_recall.png"
