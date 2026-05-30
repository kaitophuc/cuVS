from config import (
    IVFPQ_N_LISTS_SWEEP,
    IVFPQ_N_PROBES_SWEEP,
    IVFPQ_PQ_BITS_SWEEP,
    IVFPQ_PQ_DIM_SWEEP,
    IVFPQ_SWEEP_RESULTS_CSV,
)
from ivf_pq.benchmark import run_ivf_pq_configs
from support.charts import plot_ivfpq_results


def run_sweep(output_path=IVFPQ_SWEEP_RESULTS_CSV, draw_chart=True):
    configs = [
        {
            "n_lists": n_lists,
            "pq_bits": pq_bits,
            "pq_dim": pq_dim,
            "n_probes": n_probes,
        }
        for n_lists in IVFPQ_N_LISTS_SWEEP
        for pq_bits in IVFPQ_PQ_BITS_SWEEP
        for pq_dim in IVFPQ_PQ_DIM_SWEEP
        for n_probes in IVFPQ_N_PROBES_SWEEP
    ]

    print("IVF-PQ n_lists sweep:", IVFPQ_N_LISTS_SWEEP)
    print("IVF-PQ pq_bits sweep:", IVFPQ_PQ_BITS_SWEEP)
    print("IVF-PQ pq_dim sweep:", IVFPQ_PQ_DIM_SWEEP)
    print("IVF-PQ n_probes sweep:", IVFPQ_N_PROBES_SWEEP)

    results = run_ivf_pq_configs(configs, output_path)
    if draw_chart:
        plot_ivfpq_results(results)
    return results
