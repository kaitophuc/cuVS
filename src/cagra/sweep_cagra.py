import sys
from itertools import product
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC_DIR))

from config import RESULTS_DIR
from cagra.run_cagra_optimized import CAGRA_OPTIMIZED_PARAMS, run_cagra_configs


CAGRA_SWEEP_RESULTS_CSV = RESULTS_DIR / "cagra_sweep_results.csv"

BUILD_SWEEP = [
    {
        "graph_degree": 16,
        "intermediate_graph_degree": 32,
        "build_algo": "ivf_pq",
    },
    {
        "graph_degree": 32,
        "intermediate_graph_degree": 64,
        "build_algo": "ivf_pq",
    },
]

ITOPK_SIZE_SWEEP = [16, 24, 32, 48, 64]
BATCH_SIZE_SWEEP = [64, 128, 256]
ALGO_SWEEP = ["auto"]
SEARCH_WIDTH_SWEEP = [1]


def make_config(build_config, itopk_size, batch_size, algo, search_width):
    config = dict(CAGRA_OPTIMIZED_PARAMS)
    config.update(build_config)
    config.update(
        {
            "itopk_size": itopk_size,
            "max_queries": batch_size,
            "algo": algo,
            "search_width": search_width,
            "enable_exact_rerank": True,
            "rerank_candidate_k": itopk_size,
            "n_rows_per_batch": batch_size,
        }
    )
    return config


def make_sweep_configs():
    configs = []
    for build_config in BUILD_SWEEP:
        for itopk_size, batch_size, algo, search_width in product(
            ITOPK_SIZE_SWEEP,
            BATCH_SIZE_SWEEP,
            ALGO_SWEEP,
            SEARCH_WIDTH_SWEEP,
        ):
            configs.append(
                make_config(
                    build_config=build_config,
                    itopk_size=itopk_size,
                    batch_size=batch_size,
                    algo=algo,
                    search_width=search_width,
                )
            )
    return configs


def main():
    configs = make_sweep_configs()
    print(f"CAGRA build configs: {len(BUILD_SWEEP)}")
    print(f"CAGRA total sweep configs: {len(configs)}")
    print(f"CAGRA itopk sweep: {ITOPK_SIZE_SWEEP}")
    print(f"CAGRA batch-size sweep: {BATCH_SIZE_SWEEP}")
    print(f"CAGRA algo sweep: {ALGO_SWEEP}")
    print(f"CAGRA search-width sweep: {SEARCH_WIDTH_SWEEP}")
    run_cagra_configs(configs, CAGRA_SWEEP_RESULTS_CSV)


if __name__ == "__main__":
    main()
