#!/usr/bin/env python3
"""Plot final benchmark results: latency, numerics, scaling."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def configure_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": ":",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 160,
        "savefig.dpi": 200,
    })


def plot_all(records, output_dir):
    ok = [r for r in records if r.get("status") == "ok" and "bench_cycles_mean" in r]
    ok.sort(key=lambda r: r.get("params_millions", 0))

    models = [r["model"] for r in ok]
    params = [r["params_millions"] for r in ok]
    cycles = [r["bench_cycles_mean"] for r in ok]
    ms = [c / 1e6 for c in cycles]
    cosines = [r["cosine_vs_quant"] for r in ok]
    rmses = [r["rmse_vs_quant"] for r in ok]
    cvs = [r.get("bench_cycles_cv", 0) * 100 for r in ok]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)

    # 1. Latency vs params
    ax = axes[0, 0]
    ax.bar(models, ms, color="#2166ac", width=0.6)
    for i, (m, v) in enumerate(zip(models, ms)):
        ax.text(i, v + 1, f"{v:.1f} ms", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Latency [ms]")
    ax.set_title("NPU Inference Latency (Ethos-U65-256, i.MX93)")

    # 2. Cosine similarity
    ax = axes[0, 1]
    colors = ["#2166ac" if c > 0.95 else "#b2182b" for c in cosines]
    bars = ax.bar(models, cosines, color=colors, width=0.6)
    ax.set_ylabel("Cosine similarity (device vs host quantized)")
    ax.set_ylim(0.95, 1.005)
    ax.axhline(y=0.99, color="#999", linestyle="--", alpha=0.5)
    for i, v in enumerate(cosines):
        ax.text(i, v + 0.001, f"{v:.4f}", ha="center", va="bottom", fontsize=9)
    ax.set_title("Numerical Accuracy")

    # 3. Cycles scaling
    ax = axes[1, 0]
    ax.scatter(params, [c / 1e6 for c in cycles], s=80, c="#2166ac", zorder=3)
    for p, c, m in zip(params, cycles, models):
        ax.annotate(m, (p, c / 1e6), textcoords="offset points", xytext=(8, 4), fontsize=9)
    ax.set_xlabel("Parameters [millions]")
    ax.set_ylabel("NPU cycles [millions]")
    ax.set_title("Compute Scaling")

    # 4. Summary table
    ax = axes[1, 1]
    ax.axis("off")
    table_data = []
    for r in ok:
        c = r["bench_cycles_mean"]
        table_data.append([
            r["model"],
            f"{r['params_millions']}M",
            f"{c / 1e6:.1f} ms",
            f"{1000 / (c / 1e6):.1f}",
            f"{r['cosine_vs_quant']:.4f}",
            f"{r['rmse_vs_quant']:.4f}",
            f"{r.get('bench_cycles_cv', 0) * 100:.2f}%",
        ])
    table = ax.table(
        cellText=table_data,
        colLabels=["Model", "Params", "Latency", "FPS", "Cosine", "RMSE", "CV"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.5)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#2166ac")
            cell.set_text_props(color="white", weight="bold")
        else:
            cell.set_facecolor("#f0f0f0" if row % 2 == 0 else "white")
    ax.set_title("i.MX93 Ethos-U65-256 Fastpath Results", pad=20, fontsize=12, weight="bold")

    fig.savefig(output_dir / "fastpath_benchmark_results.png")
    plt.close(fig)
    print(f"Saved: {output_dir / 'fastpath_benchmark_results.png'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()

    configure_style()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = json.loads(Path(args.summary).read_text())
    plot_all(records, args.output_dir)


if __name__ == "__main__":
    main()
