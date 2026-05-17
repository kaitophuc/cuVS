import csv
from collections import defaultdict
from pathlib import Path

from config import K, RESULTS_DIR


def read_results_csv(csv_path=RESULTS_DIR / "ivfpq_sweep_results.csv"):
    csv_path = Path(csv_path)
    results = []
    int_fields = {"n_lists", "pq_bits", "pq_dim", "n_probes", "total_correct", "total_possible"}
    float_fields = {
        "build_time",
        "search_time",
        "queries_per_second",
        "latency_per_query",
        "online_median_latency_ms",
        "online_p95_latency_ms",
        "recall_at_10",
    }

    with csv_path.open(newline="", encoding="utf-8") as csv_file:
        for row in csv.DictReader(csv_file):
            for field in int_fields:
                row[field] = int(row[field])
            for field in float_fields:
                row[field] = float(row[field])
            results.append(row)

    return results


def plot_qps_to_recall(results):
    import matplotlib.pyplot as plt

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
    plt.title("IVF-PQ Throughput vs Recall Enable Exact Rerank")
    plt.yscale("log")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

    print(f"Saved IVF-PQ QPS vs recall plot to: {output_path}")


def plot_qps_to_recall_from_csv(csv_path=RESULTS_DIR / "ivfpq_sweep_results.csv"):
    results = read_results_csv(csv_path)
    plot_qps_to_recall(results)


def main():
    from sweep_ivfpq_benchmark import run_ivf_pq_sweep_benchmark

    results = run_ivf_pq_sweep_benchmark()
    plot_qps_to_recall(results)


if __name__ == "__main__":
    main()
