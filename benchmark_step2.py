import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "cuvs-matplotlib"))
os.environ.setdefault("CUVS_BENCH_IVFPQ_RERANK_BACKEND", "session")
os.environ.setdefault("CUVS_BENCH_IVFPQ_RERANK_STORAGE_DTYPE", "float16")
os.environ.setdefault("CUVS_BENCH_CAGRA_RERANK_BACKEND", "multi_gpu")
os.environ.setdefault("CUVS_BENCH_CAGRA_RERANK_STORAGE_DTYPE", "float16")

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def initialize_cuda_device():
    from cuda.bindings import runtime

    result = runtime.cudaSetDevice(0)
    if int(result[0]) != 0:
        raise RuntimeError(f"cudaSetDevice(0) failed: {result[0]}")



STEP2_DIR = ROOT_DIR / "results" / "Step2"

IVFPQ_EXACT_RERANK_CSV = STEP2_DIR / "ivfpq_exact_rerank_sweep_results.csv"
CAGRA_NO_RERANK_CSV = STEP2_DIR / "cagra_no_rerank_sweep_results.csv"
CAGRA_EXACT_RERANK_CSV = STEP2_DIR / "cagra_exact_rerank_sweep_results.csv"

IVFPQ_EXACT_RERANK_CHART = STEP2_DIR / "ivfpq_throughput_vs_recall_exact_rerank.png"
CAGRA_NO_RERANK_CHART = STEP2_DIR / "cagra_throughput_vs_recall_no_rerank.png"
CAGRA_EXACT_RERANK_CHART = STEP2_DIR / "cagra_throughput_vs_recall_exact_rerank.png"


def make_ivfpq_exact_rerank_configs():
    configs = [
        {
            "n_lists": n_lists,
            "pq_bits": pq_bits,
            "pq_dim": pq_dim,
            "n_probes": n_probes,
        }
        for n_lists in [2048, 4096]
        for pq_bits in [4, 8]
        for pq_dim in [384, 768]
        for n_probes in [16, 32, 64, 128]
    ]
    extra_configs = [
        (1024, 4, 384, [8, 16, 24, 32, 48, 64]),
        (2048, 4, 384, [40, 48, 56]),
        (4096, 4, 384, [40, 48, 56]),
        (8192, 4, 384, [16, 24, 32, 48, 64]),
    ]
    existing = {
        (config["n_lists"], config["pq_bits"], config["pq_dim"], config["n_probes"])
        for config in configs
    }
    for n_lists, pq_bits, pq_dim, n_probes_values in extra_configs:
        for n_probes in n_probes_values:
            key = (n_lists, pq_bits, pq_dim, n_probes)
            if key in existing:
                continue
            configs.append(
                {
                    "n_lists": n_lists,
                    "pq_bits": pq_bits,
                    "pq_dim": pq_dim,
                    "n_probes": n_probes,
                }
            )
            existing.add(key)
    return configs


def build_key_for_ivfpq_config(config):
    return (
        config["n_lists"],
        config["pq_bits"],
        config["pq_dim"],
    )


def make_cagra_base_config(
    build_config,
    itopk_size,
    batch_size,
    search_width,
    exact_rerank,
):
    from cagra.run_cagra_optimized import CAGRA_OPTIMIZED_PARAMS

    config = dict(CAGRA_OPTIMIZED_PARAMS)
    config.update(build_config)
    config.update(
        {
            "itopk_size": itopk_size,
            "max_queries": batch_size,
            "n_rows_per_batch": batch_size,
            "enable_exact_rerank": exact_rerank,
            "rerank_candidate_k": itopk_size if exact_rerank else 0,
            "search_width": search_width,
            "algo": "auto",
        }
    )
    return config


def make_cagra_configs(exact_rerank):
    build_sweep = [
        {
            "graph_degree": 16,
            "intermediate_graph_degree": 32,
            "build_algo": "ivf_pq",
        },
        {
            "graph_degree": 24,
            "intermediate_graph_degree": 48,
            "build_algo": "ivf_pq",
        },
        {
            "graph_degree": 32,
            "intermediate_graph_degree": 64,
            "build_algo": "ivf_pq",
        },
    ]
    itopk_sweep = [16, 24, 32, 48]
    if not exact_rerank:
        itopk_sweep.append(64)
    batch_size_sweep = [64, 128]
    search_width_sweep = [1, 2]

    configs = []
    for build_config in build_sweep:
        for itopk_size in itopk_sweep:
            for batch_size in batch_size_sweep:
                for search_width in search_width_sweep:
                    configs.append(
                        make_cagra_base_config(
                            build_config=build_config,
                            itopk_size=itopk_size,
                            batch_size=batch_size,
                            search_width=search_width,
                            exact_rerank=exact_rerank,
                        )
                    )
    if exact_rerank:
        selected_batch_256_configs = [
            (
                {
                    "graph_degree": 16,
                    "intermediate_graph_degree": 32,
                    "build_algo": "ivf_pq",
                },
                32,
            ),
            (
                {
                    "graph_degree": 32,
                    "intermediate_graph_degree": 64,
                    "build_algo": "ivf_pq",
                },
                48,
            ),
        ]
        for build_config, itopk_size in selected_batch_256_configs:
            configs.append(
                make_cagra_base_config(
                    build_config=build_config,
                    itopk_size=itopk_size,
                    batch_size=256,
                    search_width=1,
                    exact_rerank=True,
                )
            )
    return configs


def build_key_for_cagra_config(config):
    return (
        config["graph_degree"],
        config["intermediate_graph_degree"],
        config["build_algo"],
    )


def combine_csv_files(input_paths, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = None
    rows = []
    for input_path in input_paths:
        with input_path.open(newline="", encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file)
            if fieldnames is None:
                fieldnames = reader.fieldnames
            rows.extend(reader)

    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_cagra_configs_isolated(configs, output_path, only):
    isolate_itopk = only == "cagra-rerank"
    grouped = {}
    for config in configs:
        key = build_key_for_cagra_config(config)
        if isolate_itopk:
            key = (*key, config["itopk_size"])
        grouped.setdefault(key, []).append(config)

    temp_paths = []
    for key in sorted(grouped):
        graph_degree, intermediate_graph_degree, build_algo = key[:3]
        itopk_size = key[3] if isolate_itopk else None
        temp_path = (
            STEP2_DIR
            / (
                f".{output_path.stem}_{graph_degree}_{intermediate_graph_degree}_"
                f"{build_algo}{f'_{itopk_size}' if itopk_size is not None else ''}.csv"
            )
        )
        temp_paths.append(temp_path)
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--only",
            only,
            "--cagra-build-degree",
            str(graph_degree),
            "--cagra-output",
            str(temp_path),
            "--no-charts",
        ]
        if itopk_size is not None:
            command.extend(["--cagra-itopk-size", str(itopk_size)])
        print(
            "Running isolated CAGRA group: "
            f"graph_degree={graph_degree}, "
            f"intermediate_graph_degree={intermediate_graph_degree}, "
            f"itopk_size={itopk_size if itopk_size is not None else 'all'}, "
            f"configs={len(grouped[key])}"
        )
        subprocess.run(command, check=True)

    combine_csv_files(temp_paths, output_path)
    for temp_path in temp_paths:
        temp_path.unlink(missing_ok=True)
    print(f"\nSaved combined CAGRA results to: {output_path}")


def run_cagra_benchmarks(
    configs,
    output_path,
    only,
    cagra_build_degree=None,
    cagra_itopk_size=None,
):
    if cagra_build_degree is not None:
        initialize_cuda_device()
        from cagra.run_cagra_optimized import run_cagra_configs

        configs = [
            config
            for config in configs
            if config["graph_degree"] == cagra_build_degree
        ]
        if cagra_itopk_size is not None:
            configs = [
                config
                for config in configs
                if config["itopk_size"] == cagra_itopk_size
            ]
        run_cagra_configs(configs, output_path)
        return

    run_cagra_configs_isolated(configs, output_path, only)


def run_ivfpq_configs_isolated(configs, output_path):
    grouped = {}
    for config in configs:
        grouped.setdefault(build_key_for_ivfpq_config(config), []).append(config)

    temp_paths = []
    for n_lists, pq_bits, pq_dim in sorted(grouped):
        temp_path = STEP2_DIR / f".{output_path.stem}_{n_lists}_{pq_bits}_{pq_dim}.csv"
        temp_paths.append(temp_path)
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--only",
            "ivfpq",
            "--ivfpq-n-lists",
            str(n_lists),
            "--ivfpq-pq-bits",
            str(pq_bits),
            "--ivfpq-pq-dim",
            str(pq_dim),
            "--ivfpq-output",
            str(temp_path),
            "--no-charts",
        ]
        print(
            "Running isolated IVF-PQ group: "
            f"n_lists={n_lists}, pq_bits={pq_bits}, pq_dim={pq_dim}, "
            f"configs={len(grouped[(n_lists, pq_bits, pq_dim)])}"
        )
        subprocess.run(command, check=True)

    combine_csv_files(temp_paths, output_path)
    for temp_path in temp_paths:
        temp_path.unlink(missing_ok=True)
    print(f"\nSaved combined IVF-PQ results to: {output_path}")


def run_ivfpq_benchmarks(
    configs,
    output_path,
    ivfpq_n_lists=None,
    ivfpq_pq_bits=None,
    ivfpq_pq_dim=None,
):
    if ivfpq_n_lists is not None:
        initialize_cuda_device()
        from ivf_pq.benchmark import run_ivf_pq_configs

        configs = [
            config
            for config in configs
            if config["n_lists"] == ivfpq_n_lists
            and config["pq_bits"] == ivfpq_pq_bits
            and config["pq_dim"] == ivfpq_pq_dim
        ]
        run_ivf_pq_configs(configs, output_path)
        return

    run_ivfpq_configs_isolated(configs, output_path)


def run_benchmarks(args):
    only = args.only

    STEP2_DIR.mkdir(parents=True, exist_ok=True)

    if only in {"all", "ivfpq"}:
        ivfpq_configs = make_ivfpq_exact_rerank_configs()
        print(f"Step 2 IVF-PQ exact-rerank configs: {len(ivfpq_configs)}")
        run_ivfpq_benchmarks(
            ivfpq_configs,
            args.ivfpq_output or IVFPQ_EXACT_RERANK_CSV,
            args.ivfpq_n_lists,
            args.ivfpq_pq_bits,
            args.ivfpq_pq_dim,
        )

    if only in {"all", "cagra-no-rerank"}:
        cagra_no_rerank_configs = make_cagra_configs(exact_rerank=False)
        print(f"Step 2 CAGRA no-rerank configs: {len(cagra_no_rerank_configs)}")
        run_cagra_benchmarks(
            cagra_no_rerank_configs,
            args.cagra_output or CAGRA_NO_RERANK_CSV,
            "cagra-no-rerank",
            args.cagra_build_degree,
            args.cagra_itopk_size,
        )

    if only in {"all", "cagra-rerank"}:
        cagra_rerank_configs = make_cagra_configs(exact_rerank=True)
        print(f"Step 2 CAGRA exact-rerank configs: {len(cagra_rerank_configs)}")
        run_cagra_benchmarks(
            cagra_rerank_configs,
            args.cagra_output or CAGRA_EXACT_RERANK_CSV,
            "cagra-rerank",
            args.cagra_build_degree,
            args.cagra_itopk_size,
        )


def draw_charts():
    draw_grouped_chart(
        csv_path=IVFPQ_EXACT_RERANK_CSV,
        output_path=IVFPQ_EXACT_RERANK_CHART,
        title="IVF-PQ Throughput vs Recall (Exact Rerank)",
        group_fields=["n_lists", "pq_bits", "pq_dim"],
        label_fn=lambda row: (
            f"lists={row['n_lists']}, bits={row['pq_bits']}, dim={row['pq_dim']}"
        ),
    )
    draw_grouped_chart(
        csv_path=CAGRA_NO_RERANK_CSV,
        output_path=CAGRA_NO_RERANK_CHART,
        title="CAGRA Throughput vs Recall (No Rerank)",
        group_fields=["graph_degree", "intermediate_graph_degree"],
        label_fn=lambda row: (
            f"degree={row['graph_degree']}, intermediate={row['intermediate_graph_degree']}"
        ),
        connect_points=False,
    )
    draw_grouped_chart(
        csv_path=CAGRA_EXACT_RERANK_CSV,
        output_path=CAGRA_EXACT_RERANK_CHART,
        title="CAGRA Throughput vs Recall (Exact Rerank)",
        group_fields=["graph_degree", "intermediate_graph_degree"],
        label_fn=lambda row: (
            f"degree={row['graph_degree']}, intermediate={row['intermediate_graph_degree']}"
        ),
        connect_points=False,
    )


def read_rows(csv_path):
    rows = []
    with csv_path.open(newline="", encoding="utf-8") as csv_file:
        for row in csv.DictReader(csv_file):
            rows.append(row)
    return rows


def draw_grouped_chart(
    csv_path,
    output_path,
    title,
    group_fields,
    label_fn,
    connect_points=True,
):
    import matplotlib.ticker as ticker
    import matplotlib.pyplot as plt

    if not csv_path.exists():
        print(f"Skipping chart; missing {csv_path}")
        return

    rows = read_rows(csv_path)
    grouped = {}
    for row in rows:
        key = tuple(row[field] for field in group_fields)
        grouped.setdefault(key, []).append(row)

    _, axis = plt.subplots(figsize=(10, 6))
    for key in sorted(grouped):
        group_rows = sorted(grouped[key], key=lambda row: float(row["recall_at_10"]))
        recall = [float(row["recall_at_10"]) for row in group_rows]
        qps = [float(row["queries_per_second"]) for row in group_rows]
        label = label_fn(group_rows[0])
        if connect_points:
            axis.plot(
                recall,
                qps,
                marker="o",
                linewidth=1.8,
                markersize=4,
                label=label,
            )
        else:
            axis.scatter(
                recall,
                qps,
                s=34,
                alpha=0.85,
                label=label,
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    axis.set_xlabel("Recall@10")
    axis.set_ylabel("Throughput (QPS)")
    axis.set_title(title)
    axis.yaxis.set_major_formatter(ticker.FuncFormatter(lambda value, _: f"{value / 1000:.0f}k"))
    axis.yaxis.set_major_locator(ticker.MaxNLocator(nbins=9))
    axis.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    axis.grid(True, which="major", alpha=0.35)
    axis.grid(True, which="minor", axis="y", alpha=0.16)
    axis.margins(x=0.04, y=0.08)
    axis.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"Saved chart to: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Run Step 2 cuVS benchmark sweeps.")
    parser.add_argument(
        "--only",
        choices=["all", "ivfpq", "cagra-no-rerank", "cagra-rerank"],
        default="all",
        help="Benchmark subset to run before charting.",
    )
    parser.add_argument(
        "--charts-only",
        action="store_true",
        help="Regenerate PNG charts from existing Step 2 CSV files.",
    )
    parser.add_argument(
        "--no-charts",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--cagra-build-degree",
        type=int,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--cagra-itopk-size",
        type=int,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--cagra-output",
        type=Path,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--ivfpq-n-lists",
        type=int,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--ivfpq-pq-bits",
        type=int,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--ivfpq-pq-dim",
        type=int,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--ivfpq-output",
        type=Path,
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.charts_only:
        run_benchmarks(args)
    if not args.no_charts:
        draw_charts()


if __name__ == "__main__":
    main()
