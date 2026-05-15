import config


def main():
    config.IVFPQ_N_LISTS_SWEEP = [4096]
    config.IVFPQ_PQ_BITS_SWEEP = [8]
    config.IVFPQ_PQ_DIM_SWEEP = [768]
    config.IVFPQ_N_PROBES_SWEEP = [32]

    # Import after overriding config because sweep_ivfpq_benchmark imports
    # these values directly from config at module import time.
    from sweep_ivfpq_benchmark import run_ivf_pq_sweep_benchmark

    run_ivf_pq_sweep_benchmark()


if __name__ == "__main__":
    main()
