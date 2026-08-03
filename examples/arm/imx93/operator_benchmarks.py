#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess  # nosec B404 - launches trusted local tooling
import sys
from pathlib import Path

from examples.arm.imx93.export_and_verify import (
    build_export_command,
    build_export_env,
    DEFAULT_COMPILER_FLAGS,
    DEFAULT_MEMORY_MODE,
    DEFAULT_SYSTEM_CONFIG,
)
from examples.arm.imx93.supported_ops import ARM_TFLITE_CONSTRAINT_HINTS


REPO_ROOT = Path(__file__).resolve().parents[3]
MODEL_FILE = Path(__file__).with_name("operator_microbench_model.py")
BENCH_OPS = (
    "argmax",
    "cat",
    "depthwise_conv2d",
    "logistic",
    "pad",
    "prelu",
    "reshape",
    "resize_bilinear",
    "resize_nearest_neighbor",
    "slice",
    "split",
    "squeeze",
    "strided_slice",
    "transpose",
    "unpack",
)

TIME_PATTERN = re.compile(
    r"Model executed successfully\s+(\d+)\s+time\(s\)\s+in\s+([0-9.]+)\s+ms\."
)
CYCLE_PATTERN = re.compile(r"Ethos-U(?: i\.MX)? cycle counter:\s+(\d+)")
NPU_OP_PATTERN = re.compile(r"NPU operators = (\d+)")
ESTIMATED_CYCLE_PATTERN = re.compile(r"Total cycles\s+(\d+) cycles/batch")


def export_operator(
    op_name: str,
    output_dir: str | Path,
    *,
    system_config: str = DEFAULT_SYSTEM_CONFIG,
    memory_mode: str = DEFAULT_MEMORY_MODE,
    env: dict[str, str] | None = None,
    size: int | None = None,
    channels_last_4d: bool = True,
) -> tuple[Path, str]:
    output_dir = Path(output_dir) / op_name
    output_dir.mkdir(parents=True, exist_ok=True)
    command = build_export_command(
        str(MODEL_FILE),
        output_dir,
        system_config=system_config,
        memory_mode=memory_mode,
        extra_compiler_flags=DEFAULT_COMPILER_FLAGS,
        channels_last_4d=channels_last_4d,
    )
    run_env = os.environ.copy() if env is None else dict(env)
    run_env["IMX93_FASTPATH_BENCH_OP"] = op_name
    if size is not None:
        run_env["IMX93_FASTPATH_BENCH_SIZE"] = str(size)
    result = subprocess.run(  # nosec B603
        command,
        check=True,
        cwd=REPO_ROOT,
        env=build_export_env(run_env),
        text=True,
        capture_output=True,
    )
    pte_files = sorted(output_dir.glob("*.pte"))
    if not pte_files:
        raise FileNotFoundError(f"No .pte file found for {op_name}")
    return max(pte_files, key=lambda candidate: candidate.stat().st_size), (
        result.stdout + result.stderr
    )


def parse_export_output(output: str) -> dict[str, int]:
    result: dict[str, int] = {}
    npu_match = NPU_OP_PATTERN.search(output)
    if npu_match is not None:
        result["npu_operators"] = int(npu_match.group(1))
    cycle_match = ESTIMATED_CYCLE_PATTERN.search(output)
    if cycle_match is not None:
        result["estimated_cycles"] = int(cycle_match.group(1))
    return result


def trim_output(output: str, line_count: int = 20) -> str:
    lines = [line for line in output.strip().splitlines() if line.strip()]
    return "\n".join(lines[-line_count:])


def build_ssh_command(
    ssh_target: str,
    remote_command: str,
    ssh_options: list[str] | None = None,
) -> list[str]:
    command = ["ssh"]
    for option in ssh_options or []:
        command.extend(["-o", option])
    command.extend([ssh_target, remote_command])
    return command


def build_scp_command(
    local_path: str | Path,
    ssh_target: str,
    remote_path: str,
    ssh_options: list[str] | None = None,
) -> list[str]:
    command = ["scp"]
    for option in ssh_options or []:
        command.extend(["-o", option])
    command.extend([str(local_path), f"{ssh_target}:{remote_path}"])
    return command


def run_remote_command(
    ssh_target: str,
    remote_command: str,
    ssh_options: list[str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # nosec B603
        build_ssh_command(ssh_target, remote_command, ssh_options),
        check=True,
        text=True,
        capture_output=True,
    )


def copy_to_remote(
    local_path: str | Path,
    ssh_target: str,
    remote_path: str,
    ssh_options: list[str] | None = None,
) -> None:
    subprocess.run(  # nosec B603
        build_scp_command(local_path, ssh_target, remote_path, ssh_options),
        check=True,
        text=True,
        capture_output=True,
    )


def run_remote_benchmark(
    op_name: str,
    pte_path: Path,
    *,
    runner: Path,
    ssh_target: str,
    remote_dir: str,
    remote_runner_path: str | None,
    ssh_options: list[str] | None,
    num_executions: int,
) -> dict[str, float | int | str]:
    remote_runner = remote_runner_path or (
        f"{remote_dir.rstrip('/')}/{Path(runner).name}"
    )
    remote_model = f"{remote_dir.rstrip('/')}/{op_name}.pte"
    run_remote_command(
        ssh_target,
        f"mkdir -p {shlex.quote(remote_dir)}",
        ssh_options,
    )
    copy_to_remote(runner, ssh_target, remote_runner, ssh_options)
    copy_to_remote(pte_path, ssh_target, remote_model, ssh_options)
    result = run_remote_command(
        ssh_target,
        " ".join(
            [
                f"chmod +x {shlex.quote(remote_runner)}",
                "&&",
                shlex.quote(remote_runner),
                f"--model_path={shlex.quote(remote_model)}",
                f"--num_executions={num_executions}",
            ]
        ),
        ssh_options,
    )
    metrics = parse_runner_output(result.stdout + result.stderr)
    metrics["remote_runner"] = remote_runner
    return metrics


def parse_runner_output(output: str) -> dict[str, float | int]:
    result: dict[str, float | int] = {}
    time_match = TIME_PATTERN.search(output)
    if time_match is not None:
        result["executions"] = int(time_match.group(1))
        result["total_ms"] = float(time_match.group(2))
    cycle_match = CYCLE_PATTERN.search(output)
    if cycle_match is not None:
        result["cycle_counter"] = int(cycle_match.group(1))
    return result


def list_ops() -> None:
    for op in BENCH_OPS:
        print(op)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--ops", nargs="+", default=list(BENCH_OPS))
    parser.add_argument("--output", default="arm_test/models/imx93-ops")
    parser.add_argument("--system_config", default=DEFAULT_SYSTEM_CONFIG)
    parser.add_argument("--memory_mode", default=DEFAULT_MEMORY_MODE)
    parser.add_argument("--summary")
    parser.add_argument("--runner", type=Path)
    parser.add_argument("--ssh_target")
    parser.add_argument("--remote_dir", default="/tmp/imx93-fastpath-bench")
    parser.add_argument("--remote_runner_path")
    parser.add_argument("--ssh_option", action="append", default=[])
    parser.add_argument("--num_executions", type=int, default=10)
    args = parser.parse_args()

    if args.list:
        list_ops()
        return

    if bool(args.runner) != bool(args.ssh_target):
        raise SystemExit("--runner and --ssh_target must be provided together")

    results: dict[str, dict[str, object]] = {}
    for op_name in args.ops:
        try:
            pte_path, export_output = export_operator(
                op_name,
                args.output,
                system_config=args.system_config,
                memory_mode=args.memory_mode,
            )
            metrics = parse_export_output(export_output)
            results[op_name] = {
                "status": "ok",
                "pte": str(pte_path),
                "constraint_hint": ARM_TFLITE_CONSTRAINT_HINTS.get(op_name, ""),
                **metrics,
            }
            if args.runner and args.ssh_target:
                try:
                    remote_metrics = run_remote_benchmark(
                        op_name,
                        pte_path,
                        runner=args.runner,
                        ssh_target=args.ssh_target,
                        remote_dir=args.remote_dir,
                        remote_runner_path=args.remote_runner_path,
                        ssh_options=args.ssh_option,
                        num_executions=args.num_executions,
                    )
                    results[op_name].update(remote_metrics)
                except subprocess.CalledProcessError as error:
                    output = ""
                    if error.stdout:
                        output += error.stdout
                    if error.stderr:
                        output += error.stderr
                    results[op_name]["status"] = "run_failed"
                    results[op_name]["error"] = str(error)
                    results[op_name]["log_tail"] = trim_output(output)
            print(f"{op_name}: {pte_path}")
        except subprocess.CalledProcessError as error:
            output = ""
            if error.stdout:
                output += error.stdout
            if error.stderr:
                output += error.stderr
            results[op_name] = {
                "status": "export_failed",
                "constraint_hint": ARM_TFLITE_CONSTRAINT_HINTS.get(op_name, ""),
                "error": str(error),
                "log_tail": trim_output(output),
            }
            print(f"{op_name}: export_failed")

    if args.summary:
        summary_path = Path(args.summary)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w") as handle:
            json.dump(results, handle, indent=2)


if __name__ == "__main__":
    sys.exit(main())
