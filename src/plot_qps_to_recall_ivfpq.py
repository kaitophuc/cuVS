from collections import defaultdict

import matplotlib.pyplot as plt

from config import K, RESULTS_DIR
from sweep_ivfpq_benchmark import run_ivf_pq_sweep_benchmark


def plot_qps_to_recall(results):
    output_path = RESULTS_DIR / "ivfpq_qps_vs_recall_enable_exactK.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    grouped_results = defaultdict(list)
    for result in results:
        config_key = (result["n_lists"], result["pq_bits"], result["pq_dim"])
        grouped_results[config_key].append(result)

    plt.figure(figsize=(10, 6))

    for _config_key, config_results in sorted(grouped_results.items()):
        config_results = sorted(config_results, key=lambda result: result["recall_at_10"])
        recall = [result["recall_at_10"] for result in config_results]
        qps = [result["queries_per_second"] for result in config_results]

        plt.plot(
            recall,
            qps,
            marker="o",
            linewidth=1.8,
            markersize=4,
        )

    plt.xlabel(f"Recall@{K}")
    plt.ylabel("Throughput (QPS)")
    plt.title("IVF-PQ Throughput vs Recall")
    plt.yscale("log")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

    print(f"Saved IVF-PQ QPS vs recall plot to: {output_path}")


def main():
    results = run_ivf_pq_sweep_benchmark()
    plot_qps_to_recall(results)


if __name__ == "__main__":
    main()
