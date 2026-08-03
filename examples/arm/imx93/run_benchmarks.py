#!/usr/bin/env python3
"""Run proper multi-execution benchmarks on device.

Exports each model, runs N times on device, captures timing + numerics.

Usage:
    python -m examples.arm.imx93.run_benchmarks --output /tmp/fastpath-bench-final
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

_EXECUTORCH_DIR = Path(__file__).resolve().parents[3]
if str(_EXECUTORCH_DIR / "src") not in sys.path:
    sys.path.insert(0, str(_EXECUTORCH_DIR / "src"))
if str(_EXECUTORCH_DIR) not in sys.path:
    sys.path.insert(0, str(_EXECUTORCH_DIR))

from examples.arm.imx93.device_numerics_test import (
    export_and_test,
    REMOTE_DIR,
    REMOTE_RUNNER,
    scp_to,
    ssh_cmd,
)

MODELS = ["mv2", "resnet18", "resnet50", "ic3"]
MODEL_PARAMS = {"mv2": 3.5, "resnet18": 11.7, "resnet50": 25.6, "ic3": 27.2}
NUM_EXECUTIONS = 10


def parse_timing(output: str) -> dict:
    result = {}
    for line in output.splitlines():
        if "cycle counter" in line.lower():
            try:
                result["cycles"] = int(line.split(":")[-1].strip())
            except ValueError:
                pass
        m = re.search(r"Execution (\d+) returned status (0x\w+)", line)
        if m:
            result.setdefault("exec_statuses", []).append(m.group(2))
        m = re.search(r"(\d+) inference\(s\) completed.*?(\d+\.?\d*)\s*ms", line)
        if m:
            result["total_ms"] = float(m.group(2))
    return result


def run_benchmark(model_name: str, output_dir: Path, num_exec: int) -> dict:
    # First get the numerics result (single execution with full comparison)
    record = export_and_test(model_name, output_dir, channels_last=False)
    if record.get("status") != "ok":
        return record

    record["params_millions"] = MODEL_PARAMS.get(model_name)

    # Now run multiple times for timing
    pte_path = Path(record["pte_path"])
    remote_pte = f"{REMOTE_DIR}/{model_name}_bench.pte"
    remote_input = f"{REMOTE_DIR}/{model_name}_bench_input.bin"
    remote_out = f"{REMOTE_DIR}/{model_name}_bench_out"

    input_bin = output_dir / f"{model_name}_nchw" / "input-0.bin"
    scp_to(str(pte_path), remote_pte)
    scp_to(str(input_bin), remote_input)

    run_cmd = (
        f"rm -f {remote_out}-*.bin && "
        f"{REMOTE_RUNNER} "
        f"--model_path={remote_pte} "
        f"--inputs={remote_input} "
        f"--num_executions={num_exec} "
        f"--output_file={remote_out}"
    )
    print(f"  Running {num_exec} executions...")
    result = ssh_cmd(run_cmd, timeout=300)
    combined = (result.stdout or "") + (result.stderr or "")

    if result.returncode != 0:
        record["bench_status"] = "failed"
        record["bench_log"] = combined[-500:]
        return record

    # Parse all cycle counters
    cycles = []
    for line in combined.splitlines():
        if "cycle counter" in line.lower():
            try:
                cycles.append(int(line.split(":")[-1].strip()))
            except ValueError:
                pass

    # Parse total timing
    timing = parse_timing(combined)

    record["bench_executions"] = num_exec
    record["bench_cycles"] = cycles
    if cycles:
        record["bench_cycles_mean"] = float(np.mean(cycles))
        record["bench_cycles_std"] = float(np.std(cycles))
        record["bench_cycles_cv"] = (
            float(np.std(cycles) / np.mean(cycles)) if np.mean(cycles) > 0 else 0
        )
        record["bench_ms_per_inference"] = (
            float(np.mean(cycles)) / 1000.0
        )  # cycles @ 1GHz = us
    record.update(timing)

    print(
        f"  {num_exec} runs: {np.mean(cycles):.0f} +/- {np.std(cycles):.0f} cycles "
        f"(CV={record.get('bench_cycles_cv', 0):.3f})"
    )
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=MODELS)
    parser.add_argument(
        "--output", type=Path, default=Path("/tmp/fastpath-bench-final")
    )
    parser.add_argument("--num-executions", type=int, default=NUM_EXECUTIONS)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    records = []

    for model_name in args.models:
        print(f"\n{'=' * 60}")
        print(f"Benchmarking: {model_name}")
        print(f"{'=' * 60}")
        try:
            record = run_benchmark(model_name, args.output, args.num_executions)
        except Exception as e:
            record = {"model": model_name, "status": "error", "error": str(e)}
            print(f"  ERROR: {e}")
        records.append(record)

    summary_path = args.output / "summary.json"
    summary_path.write_text(json.dumps(records, indent=2))

    # Print results table
    print(f"\n{'=' * 90}")
    print(
        f"{'Model':<12} {'Params':<8} {'Top1':<5} {'Cosine':<8} {'RMSE':<8} {'Cycles':<12} {'CV%':<6} {'ms':<8}"
    )
    print("-" * 90)
    for r in records:
        if r.get("status") != "ok":
            print(f"{r['model']:<12} {'FAILED'}")
            continue
        print(
            f"{r['model']:<12} "
            f"{r.get('params_millions', '?'):<8} "
            f"{'Y' if r.get('top1_match_quant') else 'N':<5} "
            f"{r.get('cosine_vs_quant', 0):<8.4f} "
            f"{r.get('rmse_vs_quant', 0):<8.4f} "
            f"{r.get('bench_cycles_mean', 0):<12.0f} "
            f"{r.get('bench_cycles_cv', 0) * 100:<6.2f} "
            f"{r.get('bench_cycles_mean', 0) / 1000:<8.1f}"
        )

    print(f"\nSummary: {summary_path}")


if __name__ == "__main__":
    raise SystemExit(main())
