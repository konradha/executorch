#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.grid": True,
            "grid.alpha": 0.35,
            "grid.linestyle": ":",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 160,
            "savefig.dpi": 200,
            "text.usetex": False,
        }
    )


def _plot_series(
    df: pd.DataFrame,
    x_key: str,
    y_key: str,
    xlabel: str,
    ylabel: str,
    path: Path,
    *,
    xscale: str = "log",
    yscale: str = "log",
) -> None:
    fig, ax = plt.subplots(figsize=(7.6, 4.8), constrained_layout=True)
    ordered = df.sort_values(x_key)
    ax.plot(
        ordered[x_key],
        ordered[y_key],
        color="#2166ac",
        marker="o",
        linewidth=1.8,
        markersize=5,
    )
    for row in ordered.itertuples(index=False):
        ax.annotate(
            row.model,
            (getattr(row, x_key), getattr(row, y_key)),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=8,
        )
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, which="major")
    ax.grid(True, which="minor", alpha=0.18)
    fig.savefig(path)
    plt.close(fig)


def _plot_match_bars(df: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.6, 4.6), constrained_layout=True)
    ordered = df.sort_values("params_millions")
    x = range(len(ordered))
    ax.bar(
        [value - 0.18 for value in x],
        ordered["top1_match"],
        width=0.36,
        color="#67a9cf",
        label="top-1",
    )
    ax.bar(
        [value + 0.18 for value in x],
        ordered["top5_match"],
        width=0.36,
        color="#2166ac",
        label="top-5",
    )
    ax.set_xticks(list(x), ordered["model"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Match")
    ax.legend(frameon=False)
    fig.savefig(path)
    plt.close(fig)


def _plot_status(df: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.6, 3.8), constrained_layout=True)
    ordered = df.sort_values("params_millions")
    colors = ordered["status"].map({"ok": "#2166ac", "run_failed": "#b2182b"}).fillna("#ef8a62")
    values = ordered["status"].eq("ok").astype(int)
    ax.bar(ordered["model"], values, color=colors)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Run status")
    ax.set_yticks([0, 1], ["failed", "ok"])
    fig.savefig(path)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    _configure_style()
    records = json.loads(Path(args.summary).read_text())
    df = pd.DataFrame.from_records(records)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _plot_status(df, output_dir / "model_status.png")
    success_df = df[df["status"] == "ok"].copy()
    if success_df.empty:
        raise SystemExit("No successful model measurements found.")
    _plot_series(
        success_df[success_df["params_millions"].notna()],
        "params_millions",
        "total_ms",
        "Parameters [millions]",
        "Latency [ms]",
        output_dir / "latency_vs_params.png",
    )
    if "cycle_counter" in success_df.columns:
        _plot_series(
            success_df[
                success_df["params_millions"].notna()
                & success_df["cycle_counter"].notna()
            ],
            "params_millions",
            "cycle_counter",
            "Parameters [millions]",
            "Ethos-U cycle counter",
            output_dir / "cycles_vs_params.png",
        )
    _plot_series(
        success_df[success_df["pte_bytes"].notna()],
        "pte_bytes",
        "total_ms",
        "PTE size [bytes]",
        "Latency [ms]",
        output_dir / "latency_vs_pte_bytes.png",
    )
    error_df = success_df.copy()
    error_df["first10_rmse"] = error_df["first10_rmse"].clip(lower=1e-12)
    _plot_series(
        error_df[error_df["params_millions"].notna()],
        "params_millions",
        "first10_rmse",
        "Parameters [millions]",
        "First-10 RMSE",
        output_dir / "first10_rmse_vs_params.png",
        yscale="log",
    )
    _plot_match_bars(success_df, output_dir / "topk_match.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
