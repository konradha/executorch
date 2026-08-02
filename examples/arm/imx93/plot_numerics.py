#!/usr/bin/env python3
"""Plot device numerics test results."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def configure_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.grid": True,
        "grid.alpha": 0.35,
        "grid.linestyle": ":",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 160,
        "savefig.dpi": 200,
    })


def plot_cosine_and_rmse(records, output_dir):
    ok = [r for r in records if r.get("status") == "ok" and "cosine_vs_quant" in r]
    if not ok:
        return

    models = [r["model"] for r in ok]
    cosines = [r["cosine_vs_quant"] for r in ok]
    rmses = [r["rmse_vs_quant"] for r in ok]
    top1 = [r.get("top1_match_quant", 0) for r in ok]
    layouts = [r.get("layout", "?") for r in ok]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

    # Cosine similarity
    labels = [f"{m}\n({l})" for m, l in zip(models, layouts)]
    colors = ["#2166ac" if t else "#b2182b" for t in top1]
    bars = ax1.bar(labels, cosines, color=colors)
    ax1.set_ylabel("Cosine similarity (device vs host quantized)")
    ax1.set_ylim(0, 1.05)
    ax1.axhline(y=0.95, color="#999", linestyle="--", alpha=0.5, label="0.95 threshold")
    ax1.legend(frameon=False, fontsize=8)
    for bar, val in zip(bars, cosines):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    # RMSE
    bars2 = ax2.bar(labels, rmses, color=colors)
    ax2.set_ylabel("RMSE (device vs host quantized)")
    ax2.set_yscale("log")
    for bar, val in zip(bars2, rmses):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.1,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    fig.suptitle("Device vs Host Quantized Numerics (blue=top-1 match, red=mismatch)")
    fig.savefig(output_dir / "numerics_cosine_rmse.png")
    plt.close(fig)


def plot_ab_comparison(records, output_dir):
    """Plot channels_last vs NCHW comparison if both are present."""
    models_seen = {}
    for r in records:
        if r.get("status") != "ok" or "cosine_vs_quant" not in r:
            continue
        key = r["model"]
        if key not in models_seen:
            models_seen[key] = {}
        models_seen[key][r.get("layout", "?")] = r

    both = {m: v for m, v in models_seen.items() if "clast" in v and "nchw" in v}
    if not both:
        return

    models = sorted(both.keys())
    x = np.arange(len(models))
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

    nchw_cos = [both[m]["nchw"]["cosine_vs_quant"] for m in models]
    clast_cos = [both[m]["clast"]["cosine_vs_quant"] for m in models]
    ax1.bar(x - width / 2, nchw_cos, width, label="NCHW", color="#2166ac")
    ax1.bar(x + width / 2, clast_cos, width, label="channels_last", color="#b2182b")
    ax1.set_ylabel("Cosine similarity")
    ax1.set_ylim(0, 1.1)
    ax1.set_xticks(x, models)
    ax1.legend(frameon=False)
    ax1.set_title("Cosine similarity: device vs host quantized")

    nchw_rmse = [both[m]["nchw"]["rmse_vs_quant"] for m in models]
    clast_rmse = [both[m]["clast"]["rmse_vs_quant"] for m in models]
    ax2.bar(x - width / 2, nchw_rmse, width, label="NCHW", color="#2166ac")
    ax2.bar(x + width / 2, clast_rmse, width, label="channels_last", color="#b2182b")
    ax2.set_ylabel("RMSE")
    ax2.set_xticks(x, models)
    ax2.legend(frameon=False)
    ax2.set_title("RMSE: device vs host quantized")

    fig.suptitle("channels_last breaks device numerics on i.MX93 Ethos-U65")
    fig.savefig(output_dir / "nchw_vs_clast.png")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", nargs="+", required=True, help="summary.json files")
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()

    configure_style()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_records = []
    for path in args.summary:
        all_records.extend(json.loads(Path(path).read_text()))

    plot_cosine_and_rmse(all_records, args.output_dir)
    plot_ab_comparison(all_records, args.output_dir)
    print(f"Plots saved to {args.output_dir}")


if __name__ == "__main__":
    main()
