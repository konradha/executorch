#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.colors as mcolors
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


def _plot_metric(
    df: pd.DataFrame,
    metric: str,
    ylabel: str,
    path: Path,
    yscale: str,
) -> None:
    fig, ax = plt.subplots(figsize=(7.6, 4.8), constrained_layout=True)
    groups = list(df.groupby("op"))
    colors = plt.get_cmap("tab10")(range(max(len(groups), 1)))
    for color, (op_name, group) in zip(colors, groups, strict=True):
        ordered = group.sort_values("input_elements")
        ax.plot(
            ordered["input_elements"],
            ordered[metric],
            marker="o",
            linewidth=1.8,
            markersize=4.5,
            color=color,
            label=op_name,
        )
    ax.set_xscale("log", base=2)
    ax.set_yscale(yscale)
    ax.set_xlabel("Input elements")
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False, ncol=2)
    ax.grid(True, which="major")
    ax.grid(True, which="minor", alpha=0.18)
    fig.savefig(path)
    plt.close(fig)


def _plot_status_matrix(df: pd.DataFrame, path: Path) -> None:
    order = sorted(df["op"].unique())
    status_to_value = {
        "ok": 2,
        "run_failed": 1,
        "export_failed": 0,
    }
    pivot = (
        df.assign(value=lambda frame: frame["status"].map(status_to_value).fillna(-1))
        .pivot(index="op", columns="size", values="value")
        .reindex(order)
        .sort_index(axis=1)
        .fillna(-1)
    )
    fig, ax = plt.subplots(
        figsize=(7.6, 0.55 * len(order) + 1.4), constrained_layout=True
    )
    cmap = mcolors.ListedColormap(["#b2182b", "#ef8a62", "#67a9cf", "#2166ac"])
    norm = mcolors.BoundaryNorm([-1.5, -0.5, 0.5, 1.5, 2.5], cmap.N)
    image = ax.imshow(
        pivot.to_numpy(),
        cmap=cmap,
        norm=norm,
        aspect="auto",
        interpolation="nearest",
    )
    ax.set_yticks(range(len(pivot.index)), labels=pivot.index)
    ax.set_xticks(
        range(len(pivot.columns)), labels=[str(size) for size in pivot.columns]
    )
    ax.set_xlabel("Operator size parameter")
    ax.set_ylabel("Operator")
    colorbar = fig.colorbar(image, ax=ax, shrink=0.9)
    colorbar.set_ticks([0, 1, 2], labels=["export failed", "run failed", "ok"])
    for row, op_name in enumerate(pivot.index):
        for col, size in enumerate(pivot.columns):
            value = int(pivot.loc[op_name, size])
            label = {2: "ok", 1: "run", 0: "exp", -1: "n/a"}[value]
            ax.text(
                col,
                row,
                label,
                ha="center",
                va="center",
                color="white" if value <= 0 else "black",
                fontsize=8,
            )
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
    success_df = df[df["status"] == "ok"].copy()
    if success_df.empty:
        raise SystemExit("No successful measurements found.")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _plot_status_matrix(df, output_dir / "status_matrix.png")
    _plot_metric(
        success_df[success_df["total_ms"].notna()],
        "total_ms",
        "Latency [ms]",
        output_dir / "latency_vs_input_elements.png",
        "log",
    )
    if "cycle_counter" in success_df.columns:
        _plot_metric(
            success_df[success_df["cycle_counter"].notna()],
            "cycle_counter",
            "Ethos-U cycle counter",
            output_dir / "cycles_vs_input_elements.png",
            "log",
        )
    precision = success_df[success_df["max_abs_error"].notna()].copy()
    precision["max_abs_error"] = precision["max_abs_error"].clip(lower=1e-9)
    _plot_metric(
        precision,
        "max_abs_error",
        "Max absolute error",
        output_dir / "max_abs_error_vs_input_elements.png",
        "log",
    )
    rmse = success_df[success_df["rmse"].notna()].copy()
    rmse["rmse"] = rmse["rmse"].clip(lower=1e-12)
    _plot_metric(
        rmse,
        "rmse",
        "RMSE",
        output_dir / "rmse_vs_input_elements.png",
        "log",
    )
    relative_rmse = success_df[success_df["rmse_over_range"].notna()].copy()
    relative_rmse["rmse_over_range"] = relative_rmse["rmse_over_range"].clip(lower=1e-9)
    _plot_metric(
        relative_rmse,
        "rmse_over_range",
        "RMSE / output range",
        output_dir / "relative_rmse_vs_input_elements.png",
        "log",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
