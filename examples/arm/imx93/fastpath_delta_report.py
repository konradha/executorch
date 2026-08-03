#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

_EXECUTORCH_DIR = Path(__file__).resolve().parents[3]
_EXECUTORCH_DIR_STR = str(_EXECUTORCH_DIR)
if _EXECUTORCH_DIR_STR not in sys.path:
    sys.path.insert(0, _EXECUTORCH_DIR_STR)

from examples.arm.imx93.supported_ops import (
    IMX93_FASTPATH_MATRIX,
    MODEL_PROVEN_CORE_OPS,
    trusted_delegated_ops,
    unresolved_ops,
)


STATUS_TITLES = {
    "correct": "Clean delegated on i.MX93",
    "metadata_only": "Officially supported but metadata-only here",
    "not_delegated": "Officially supported but not delegated here",
    "export_failed": "Officially supported but blocked at export",
    "unsafe_for_inference": "Delegated but not inference-safe on i.MX93",
}

GAP_LAYER_TITLES = {
    "none": "No observed gap",
    "metadata_only": "Metadata-only op",
    "executorch_lowering": "ExecuTorch lowering/partition gap",
    "vela_compile": "Vela compile-time gap",
    "imx_runtime": "i.MX93 runtime gap",
}


def classify_delta() -> dict[str, list[tuple[str, dict[str, object]]]]:
    grouped: dict[str, list[tuple[str, dict[str, object]]]] = defaultdict(list)
    for op_name, row in IMX93_FASTPATH_MATRIX.items():
        grouped[str(row["observed_status"])].append((op_name, row))
    return dict(grouped)


def render_markdown() -> str:
    grouped = classify_delta()
    trusted = trusted_delegated_ops()
    unresolved = unresolved_ops()
    lines = [
        "# i.MX93 Fastpath Delta",
        "",
        "This report compares the under-documented U65/TFLite operator set against",
        "what the ExecuTorch i.MX93 Linux fastpath actually delivers today.",
        "It does not describe the full model-proven CNN core by itself.",
        "",
        "Source basis:",
        "",
        "- Official support and constraints come from the Vela 4.5.0 `--supported-ops-report` for U55/U65.",
        "- Fastpath status comes from this branch's export checks, delegated payload inspection, and on-device runner validation.",
        "- The broader CNN core path is anchored by full-model delegation artifacts from MobileNet/ResNet-style exports.",
        "",
        "Status meanings:",
        "",
        "- `correct`: emits a delegated payload and matches device execution.",
        "- `metadata_only`: supported as a view/bookkeeping op, not meaningful NPU work.",
        "- `not_delegated`: officially supported in principle, but no delegated payload on this path.",
        "- `export_failed`: blocked before device execution.",
        "- `unsafe_for_inference`: delegates, but the i.MX93 runtime path is still wrong or unstable.",
        "",
        "## Model-Proven Core Path",
        "",
        "These operators are already proven by heavily or fully delegated CNN-class",
        "models in the sibling `executorch/` checkout. This is the main performance",
        "path for practical inference on i.MX93.",
        "",
        ", ".join(f"`{op_name}`" for op_name in MODEL_PROVEN_CORE_OPS),
        "",
        "## Trusted Delegated Set",
        "",
        "These are the under-documented ops that are both officially supported on",
        "U65 and currently validated on the i.MX93 Linux fastpath.",
        "",
        ", ".join(f"`{op_name}`" for op_name in trusted),
        "",
        "## Open Delta Set",
        "",
        "These ops are officially supported on paper but still have a real gap on",
        "the current fastpath, or are metadata-only and should not be counted as",
        "delegated NPU work.",
        "",
        ", ".join(f"`{op_name}`" for op_name in unresolved),
        "",
    ]
    ordered_statuses = (
        "correct",
        "metadata_only",
        "not_delegated",
        "export_failed",
        "unsafe_for_inference",
    )
    for status in ordered_statuses:
        rows = grouped.get(status, [])
        if not rows:
            continue
        lines.append(f"## {STATUS_TITLES[status]}")
        lines.append("")
        lines.append("| Op | TFLite op | Sizes | Payload | Gap layer | Notes |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for op_name, row in sorted(rows):
            sizes = ", ".join(str(size) for size in row["observed_sizes"])
            payload = "yes" if row["delegated_payload"] else "no"
            gap_layer = GAP_LAYER_TITLES[str(row["gap_layer"])]
            notes = " ".join(str(row["observed_notes"]).split())
            lines.append(
                f"| `{op_name}` | `{row['tflite_op']}` | `{sizes}` | {payload} | {gap_layer} | {notes} |"
            )
        lines.append("")
    lines.append("## Constraint Boundaries For The Open Delta Set")
    lines.append("")
    for op_name in unresolved:
        row = IMX93_FASTPATH_MATRIX[op_name]
        lines.append(f"### `{op_name}`")
        lines.append("")
        lines.append(f"- TFLite op: `{row['tflite_op']}`")
        lines.append(
            f"- Payload emitted: {'yes' if row['delegated_payload'] else 'no'}"
        )
        lines.append(f"- Gap layer: {GAP_LAYER_TITLES[str(row['gap_layer'])]}")
        lines.append(f"- Model guidance: {row['model_guidance']}")
        lines.append(
            f"- Fastpath outcome: {' '.join(str(row['observed_notes']).split())}"
        )
        lines.append("- Official U55/U65 constraints:")
        for constraint in row["official_constraints"]:
            lines.append(f"  - {constraint}")
        lines.append("")
    lines.append("## Delta Summary")
    lines.append("")
    lines.append("| Category | Count |")
    lines.append("| --- | ---: |")
    for status in ordered_statuses:
        lines.append(f"| {STATUS_TITLES[status]} | {len(grouped.get(status, []))} |")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    print(render_markdown())


if __name__ == "__main__":
    main()
