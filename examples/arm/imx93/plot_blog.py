#!/usr/bin/env python3
"""Two publication-quality plots for the blogpost."""

import json
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": ":",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 180,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
    }
)

OUT = Path("/tmp/fastpath-plots")
OUT.mkdir(exist_ok=True)

# ── Plot 1: Model dashboard ─────────────────────────────────────────────

models = ["mv2", "resnet18", "ic3", "resnet50", "mv3"]
ms = [8.7, 22.8, 34.8, 100.5, None]
cosine = [0.989, 0.999, 0.987, 0.999, None]
status = ["ok", "ok", "ok", "ok", "firmware fail"]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw={"wspace": 0.4})

# Latency
colors1 = ["#2166ac" if s == "ok" else "#b2182b" for s in status]
bars_ms = [v if v else 0 for v in ms]
ax1.barh(models, bars_ms, color=colors1, height=0.6, zorder=3)
for i, value in enumerate(ms):
    if value:
        ax1.text(value + 2, i, f"{value:.1f} ms", va="center", fontsize=10)
    else:
        ax1.text(
            2,
            i,
            "IOCTL failure (NXP firmware)",
            va="center",
            fontsize=9,
            color="#b2182b",
            style="italic",
        )
ax1.set_xlabel("T/inference / [ms]")
ax1.set_xlim(0, 130)
ax1.invert_yaxis()

ax1.set_yticks(range(len(models)))
ax1.set_yticklabels(models, fontsize=10)

# Cosine
colors2 = ["#2166ac" if s == "ok" else "#b2182b" for s in status]
bars_cos = [v if v else 0 for v in cosine]
ax2.barh(models, bars_cos, color=colors2, height=0.6, zorder=3)
for i, value in enumerate(cosine):
    if value:
        ax2.text(value + 0.001, i, f"{value:.3f}", va="center", fontsize=10)
    else:
        ax2.text(
            0.15, i, "broken", va="center", fontsize=9, color="#b2182b", style="italic"
        )
ax2.set_xlabel("Cosine similarity host <> device")
ax2.set_xlim(0, 1.06)
ax2.invert_yaxis()

ax2.set_yticks(range(len(models)))
ax2.set_yticklabels(models, fontsize=10)

fig.savefig(OUT / "blog_models.png")
plt.close(fig)
print(f"Saved {OUT / 'blog_models.png'}")


# ── Plot 2: Operator sweep heatmap ──────────────────────────────────────

ops_data = json.loads(Path("/tmp/op-sweep-full/sweep_summary.json").read_text())

size8_path = Path("/tmp/op-sweep-test3/sweep_summary.json")
if size8_path.exists():
    ops_data += json.loads(size8_path.read_text())

all_ops = sorted(set(r["op"] for r in ops_data))
sizes = [16, 32]

matrix = {}
for r in ops_data:
    op = r["op"]
    sz = r["size"]
    if sz in sizes:
        matrix.setdefault(op, {})[sz] = r


def sort_key(op):
    recs = matrix.get(op, {})
    cos_vals = [
        recs[s].get("cosine", 0)
        for s in sizes
        if s in recs and recs[s].get("status") == "ok"
    ]
    if cos_vals:
        return (0, -np.mean(cos_vals))
    return (1, op)


sorted_ops = sorted(matrix.keys(), key=sort_key)

fig, ax = plt.subplots(figsize=(7, 11), constrained_layout=True)

cell_h = 0.8
cell_w = 1.0
gap = 0.15

for row, op in enumerate(sorted_ops):
    y = len(sorted_ops) - 1 - row
    for col, sz in enumerate(sizes):
        x = col * (cell_w + gap)
        r = matrix.get(op, {}).get(sz)
        if r is None:
            color = "#e0e0e0"
            label = "\u2014"
        elif r.get("status") == "ok":
            cos = r.get("cosine", 0)
            intensity = max(0, min(1, (cos - 0.9) / 0.1))
            color = plt.cm.RdYlGn(0.5 + intensity * 0.5)
            label = f"{cos:.3f}"
        elif r.get("status") == "not_delegated":
            color = "#d9d9d9"
            label = "no NPU"
        elif r.get("status") == "numerics_wrong":
            cos = r.get("cosine", 0)
            color = "#f4a582"
            label = f"{cos:.2f}"
        else:
            color = "#d6604d"
            label = r.get("status", "?")[:6]

        rect = mpatches.FancyBboxPatch(
            (x, y),
            cell_w,
            cell_h,
            boxstyle="round,pad=0.05",
            facecolor=color,
            edgecolor="white",
            linewidth=1.5,
        )
        ax.add_patch(rect)
        ax.text(
            x + cell_w / 2,
            y + cell_h / 2,
            label,
            ha="center",
            va="center",
            fontsize=8,
            color="white" if label == "no NPU" else "black",
        )

    ax.text(-0.15, y + cell_h / 2, op, ha="right", va="center", fontsize=9)

# Column headers
for col, sz in enumerate(sizes):
    x = col * (cell_w + gap)
    ax.text(
        x + cell_w / 2,
        len(sorted_ops) + 0.1,
        f"size {sz}",
        ha="center",
        va="bottom",
        fontsize=11,
    )

ax.set_xlim(-0.2, len(sizes) * (cell_w + gap))
ax.set_ylim(-0.3, len(sorted_ops) + 0.6)
ax.axis("off")
ax.set_title(
    "green = correct   orange = wrong   gray = not delegated",
    fontsize=10,
    pad=12,
    color="#555",
)

fig.savefig(OUT / "blog_ops.png")
plt.close(fig)
print(f"Saved {OUT / 'blog_ops.png'}")
