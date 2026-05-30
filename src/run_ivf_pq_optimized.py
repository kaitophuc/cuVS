from config import IVFPQ_OPTIMIZED_PARAMS, IVFPQ_OPTIMIZED_RESULTS_CSV
from ivf_pq.benchmark import run_ivf_pq_configs


def main():
    """Run the current best single IVF-PQ configuration."""
    run_ivf_pq_configs([IVFPQ_OPTIMIZED_PARAMS], IVFPQ_OPTIMIZED_RESULTS_CSV)


if __name__ == "__main__":
    main()
