import csv
import os
from collections import defaultdict
from pathlib import Path

from config import (
    IVF_COMPARISON_CHART_PATH,
    IVF_FLAT_CHART_PATH,
    IVF_FLAT_RESULTS_CSV,
    IVFPQ_CHART_PATH,
    IVFPQ_SWEEP_RESULTS_CSV,
    K,
)

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "cuvs-matplotlib"))


def read_results_csv(csv_path):
    csv_path = Path(csv_path)
    rows = []

    int_fields = {
        "n_lists",
        "n_probes",
        "pq_bits",
        "pq_dim",
        "total_correct",
        "total_possible",
    }
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
            for field in int_fields.intersection(row):
                row[field] = int(row[field])
            for field in float_fields.intersection(row):
                row[field] = float(row[field])
            rows.append(row)

    return rows


def plot_ivf_flat_results(results, output_path=IVF_FLAT_CHART_PATH):
    _plot_grouped_results(
        results=results,
        output_path=output_path,
        title="IVF-Flat Throughput vs Recall",
        group_label_fn=lambda row: f"n_lists={row['n_lists']}",
        group_key_fn=lambda row: row["n_lists"],
    )


def plot_ivfpq_results(results, output_path=IVFPQ_CHART_PATH):
    _plot_grouped_results(
        results=results,
        output_path=output_path,
        title="IVF-PQ Throughput vs Recall",
        group_label_fn=lambda row: (
            f"lists={row['n_lists']}, bits={row['pq_bits']}, dim={row['pq_dim']}"
        ),
        group_key_fn=lambda row: (row["n_lists"], row["pq_bits"], row["pq_dim"]),
    )


def draw_all_charts():
    if IVF_FLAT_RESULTS_CSV.exists():
        flat_results = read_results_csv(IVF_FLAT_RESULTS_CSV)
        plot_ivf_flat_results(flat_results)
    else:
        flat_results = []
        print(f"Skipping IVF-Flat chart; missing {IVF_FLAT_RESULTS_CSV}")

    if IVFPQ_SWEEP_RESULTS_CSV.exists():
        pq_results = read_results_csv(IVFPQ_SWEEP_RESULTS_CSV)
        plot_ivfpq_results(pq_results)
    else:
        pq_results = []
        print(f"Skipping IVF-PQ chart; missing {IVFPQ_SWEEP_RESULTS_CSV}")

    if flat_results and pq_results:
        plot_comparison(flat_results, pq_results)


def plot_comparison(
    flat_results,
    pq_results,
    output_path=IVF_COMPARISON_CHART_PATH,
):
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 6))

    _plot_scatter_series(flat_results, "IVF-Flat", marker="o")
    _plot_scatter_series(pq_results, "IVF-PQ", marker="s")

    plt.xlabel(f"Recall@{K}")
    plt.ylabel("Throughput (QPS)")
    plt.title("IVF-Flat vs IVF-PQ Throughput and Recall")
    plt.yscale("log")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"Saved comparison chart to: {output_path}")


def _plot_grouped_results(results, output_path, title, group_label_fn, group_key_fn):
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    grouped_results = defaultdict(list)

    for row in results:
        grouped_results[group_key_fn(row)].append(row)

    plt.figure(figsize=(10, 6))
    for _, group_rows in sorted(grouped_results.items()):
        group_rows = sorted(group_rows, key=lambda row: row["recall_at_10"])
        recall = [row["recall_at_10"] for row in group_rows]
        qps = [row["queries_per_second"] for row in group_rows]
        plt.plot(
            recall,
            qps,
            marker="o",
            linewidth=1.8,
            markersize=4,
            label=group_label_fn(group_rows[0]),
        )

    plt.xlabel(f"Recall@{K}")
    plt.ylabel("Throughput (QPS)")
    plt.title(title)
    plt.yscale("log")
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"Saved chart to: {output_path}")


def _plot_scatter_series(results, label, marker):
    import matplotlib.pyplot as plt

    recall = [row["recall_at_10"] for row in results]
    qps = [row["queries_per_second"] for row in results]
    plt.scatter(recall, qps, label=label, marker=marker, alpha=0.8)
