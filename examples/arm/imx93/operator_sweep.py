#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import argparse
import ctypes
import json
import os
import shlex
import subprocess  # nosec B404 - launches trusted local tooling
import sys
import time
from pathlib import Path

_EXECUTORCH_DIR = Path(__file__).resolve().parents[3]
_EXECUTORCH_DIR_STR = str(_EXECUTORCH_DIR)
if _EXECUTORCH_DIR_STR not in sys.path:
    sys.path.insert(0, _EXECUTORCH_DIR_STR)

import numpy as np
import torch
from torch import export as torch_export

from backends.arm.test.runner_utils import TosaReferenceModelDispatch
from examples.arm import aot_arm_compiler as arm_aot_compiler
from examples.arm.imx93.operator_benchmarks import (
    BENCH_OPS,
    build_ssh_command,
    export_operator,
    parse_export_output,
    parse_runner_output,
    trim_output,
)
from examples.arm.imx93.export_and_verify import (
    DEFAULT_COMPILER_FLAGS,
    DEFAULT_MEMORY_MODE,
    DEFAULT_SYSTEM_CONFIG,
    DEFAULT_TARGET,
    strip_export_guards,
)
from examples.arm.imx93.operator_microbench_model import build_case, execution_inputs
from examples.arm.imx93.supported_ops import ARM_TFLITE_CONSTRAINT_HINTS


DEFAULT_SWEEP_OPS = BENCH_OPS
DEFAULT_SIZES = (8, 16, 32, 64)
REFERENCE_TARGET = "TOSA-1.0+INT"
TRANSFER_RETRIES = 3
TRANSFER_TIMEOUT_SECONDS = 45
_STAGED_REMOTE_RUNNERS: set[tuple[str, tuple[str, ...], str]] = set()


def _tensor_sequence(output) -> list:
    if isinstance(output, tuple):
        return list(output)
    if isinstance(output, list):
        return output
    return [output]


def _case_uses_channels_last_4d(op_name: str, size: int) -> bool:
    _, example_inputs = build_case(op_name, size)
    return any(tensor.dim() == 4 for tensor in execution_inputs(example_inputs))


def _scp_with_options(
    source: str,
    destination: str,
    ssh_options: list[str],
) -> None:
    command = ["scp"]
    for option in ssh_options:
        command.extend(["-o", option])
    command.extend([source, destination])
    _run_with_retries(command)


def _run_with_retries(command: list[str]) -> subprocess.CompletedProcess[str]:
    last_error = None
    for attempt in range(TRANSFER_RETRIES):
        try:
            return subprocess.run(
                command,
                check=True,
                text=True,
                capture_output=True,
                timeout=TRANSFER_TIMEOUT_SECONDS,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
            last_error = error
            if attempt + 1 == TRANSFER_RETRIES:
                raise
            time.sleep(1 << attempt)
    raise last_error


def _compare_outputs(host_outputs, local_prefix: Path) -> dict[str, float | int]:
    host_tensors = _tensor_sequence(host_outputs)
    max_abs_error = 0.0
    rmse_sum = 0.0
    value_count = 0
    host_min = None
    host_max = None
    host_sum = 0.0
    host_sq_sum = 0.0
    for index, tensor in enumerate(host_tensors):
        host_tensor = _comparison_tensor(tensor.detach().cpu())
        host = host_tensor.numpy()
        local_file = Path(f"{local_prefix}-{index}.bin")
        device = _load_device_tensor(local_file, host_tensor)
        diff = device.astype(np.float64) - host.astype(np.float64)
        max_abs_error = max(max_abs_error, float(np.max(np.abs(diff))))
        rmse_sum += float(np.sum(diff * diff))
        value_count += int(diff.size)
        host_min = float(np.min(host)) if host_min is None else min(host_min, float(np.min(host)))
        host_max = float(np.max(host)) if host_max is None else max(host_max, float(np.max(host)))
        host_float = host.astype(np.float64)
        host_sum += float(np.sum(host_float))
        host_sq_sum += float(np.sum(host_float * host_float))
    rmse = (rmse_sum / max(value_count, 1)) ** 0.5
    host_mean = host_sum / max(value_count, 1)
    host_var = max(host_sq_sum / max(value_count, 1) - host_mean * host_mean, 0.0)
    host_std = host_var**0.5
    host_range = (host_max - host_min) if host_min is not None and host_max is not None else 0.0
    return {
        "output_tensors": len(host_tensors),
        "output_elements": value_count,
        "max_abs_error": max_abs_error,
        "rmse": rmse,
        "host_min": host_min,
        "host_max": host_max,
        "host_range": host_range,
        "host_std": host_std,
        "rmse_over_range": rmse / host_range if host_range > 0 else 0.0,
        "rmse_over_std": rmse / host_std if host_std > 0 else 0.0,
    }


def _comparison_tensor(tensor: torch.Tensor) -> torch.Tensor:
    tensor = tensor.detach().cpu()
    if tensor.dim() == 4 and tensor.is_contiguous(memory_format=torch.channels_last):
        return tensor
    if tensor.is_contiguous():
        return tensor
    return tensor.contiguous()


def _save_tensor_sequence(tensors, directory: Path, stem: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for index, tensor in enumerate(_tensor_sequence(tensors)):
        np.save(directory / f"{stem}-{index}.npy", tensor.detach().cpu().numpy())


def _tensor_storage_bytes(tensor: torch.Tensor) -> bytes:
    tensor = tensor.detach().cpu()
    return ctypes.string_at(tensor.data_ptr(), tensor.untyped_storage().nbytes())


def _load_device_tensor(local_file: Path, host_tensor: torch.Tensor) -> np.ndarray:
    host_tensor = host_tensor.detach().cpu()
    host = host_tensor.numpy()
    element_count = int(host.size)
    file_size = local_file.stat().st_size
    if element_count == 0 or file_size % max(element_count, 1) != 0:
        raise ValueError(
            f"Unexpected capture size for {local_file}: {file_size} bytes "
            f"for {element_count} elements"
        )
    bytes_per_element = file_size // max(element_count, 1)
    if host.dtype.kind in ("i", "u"):
        if bytes_per_element == 1:
            device_dtype = np.int8 if host.dtype.kind == "i" else np.uint8
        elif bytes_per_element == 2:
            device_dtype = np.int16 if host.dtype.kind == "i" else np.uint16
        elif bytes_per_element == 4:
            device_dtype = np.int32 if host.dtype.kind == "i" else np.uint32
        elif bytes_per_element == 8:
            device_dtype = np.int64 if host.dtype.kind == "i" else np.uint64
        else:
            raise ValueError(
                f"Unsupported integer capture width {bytes_per_element} "
                f"for {local_file}"
            )
    elif host.dtype.kind == "f":
        if bytes_per_element == 2:
            device_dtype = np.float16
        elif bytes_per_element == 4:
            device_dtype = np.float32
        elif bytes_per_element == 8:
            device_dtype = np.float64
        else:
            raise ValueError(
                f"Unsupported floating-point capture width {bytes_per_element} "
                f"for {local_file}"
            )
    elif host.dtype.kind == "b":
        if bytes_per_element != 1:
            raise ValueError(
                f"Unsupported boolean capture width {bytes_per_element} "
                f"for {local_file}"
            )
        device_dtype = np.bool_
    else:
        raise ValueError(f"Unsupported host dtype {host.dtype} for {local_file}")
    raw = local_file.read_bytes()
    itemsize = np.dtype(device_dtype).itemsize
    strides = tuple(int(stride) * itemsize for stride in host_tensor.stride())
    return np.ndarray(shape=host.shape, dtype=device_dtype, buffer=raw, strides=strides).copy()


def _ensure_remote_runner(
    runner: Path,
    ssh_target: str,
    ssh_options: list[str],
    remote_dir: str,
) -> str:
    remote_runner = f"{remote_dir.rstrip('/')}/{Path(runner).name}"
    cache_key = (ssh_target, tuple(ssh_options), remote_runner)
    if cache_key in _STAGED_REMOTE_RUNNERS:
        return remote_runner
    _run_with_retries(
        build_ssh_command(
            ssh_target,
            f"mkdir -p {shlex.quote(remote_dir)}",
            ssh_options,
        )
    )
    _scp_with_options(str(runner), f"{ssh_target}:{remote_runner}", ssh_options)
    _STAGED_REMOTE_RUNNERS.add(cache_key)
    return remote_runner


def _run_remote_capture(
    *,
    op_name: str,
    size: int,
    pte_path: Path,
    runner: Path,
    ssh_target: str,
    ssh_options: list[str],
    remote_dir: str,
    local_capture_dir: Path,
    inputs: tuple[torch.Tensor, ...],
    num_executions: int,
) -> tuple[dict[str, float | int | str], Path]:
    remote_runner = _ensure_remote_runner(runner, ssh_target, ssh_options, remote_dir)
    remote_model = f"{remote_dir.rstrip('/')}/{op_name}-{size}.pte"
    remote_prefix = f"{remote_dir.rstrip('/')}/{op_name}-{size}-out"
    local_prefix = local_capture_dir / f"{op_name}-{size}-out"
    _scp_with_options(str(pte_path), f"{ssh_target}:{remote_model}", ssh_options)
    local_input_dir = local_capture_dir.parent / "inputs"
    local_input_dir.mkdir(parents=True, exist_ok=True)
    remote_inputs = []
    for index, tensor in enumerate(inputs):
        local_input_path = local_input_dir / f"{op_name}-{size}-input-{index}.bin"
        local_input_path.write_bytes(_tensor_storage_bytes(tensor))
        remote_input_path = (
            f"{remote_dir.rstrip('/')}/{op_name}-{size}-input-{index}.bin"
        )
        _scp_with_options(str(local_input_path), f"{ssh_target}:{remote_input_path}", ssh_options)
        remote_inputs.append(remote_input_path)
    remote_command = " ".join(
        [
            f"rm -f {shlex.quote(remote_prefix)}-*.bin",
            "&&",
            f"chmod +x {shlex.quote(remote_runner)}",
            "&&",
            shlex.quote(remote_runner),
            f"--model_path={shlex.quote(remote_model)}",
            f"--inputs={shlex.quote(','.join(remote_inputs))}",
            f"--num_executions={num_executions}",
            f"--output_file={shlex.quote(remote_prefix)}",
        ]
    )
    result = _run_with_retries(build_ssh_command(ssh_target, remote_command, ssh_options))
    remote_list = _run_with_retries(
        build_ssh_command(
            ssh_target,
            f"ls {shlex.quote(remote_prefix)}-*.bin",
            ssh_options,
        )
    )
    local_capture_dir.mkdir(parents=True, exist_ok=True)
    for remote_file in remote_list.stdout.split():
        _scp_with_options(
            f"{ssh_target}:{remote_file}",
            str(local_capture_dir / Path(remote_file).name),
            ssh_options,
        )
    metrics = parse_runner_output(result.stdout + result.stderr)
    metrics["remote_runner"] = remote_runner
    return metrics, local_prefix


def _measure_case(
    *,
    op_name: str,
    size: int,
    output_dir: Path,
    runner: Path,
    ssh_target: str,
    ssh_options: list[str],
    remote_dir: str,
    num_executions: int,
) -> dict[str, object]:
    record: dict[str, object] = {
        "op": op_name,
        "op_name": op_name,
        "size": size,
        "constraint_hint": ARM_TFLITE_CONSTRAINT_HINTS.get(op_name, ""),
    }
    case_dir = output_dir / f"{op_name}_size{size}"
    case_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["IMX93_FASTPATH_BENCH_SIZE"] = str(size)
    try:
        pte_path, export_output = export_operator(
            op_name,
            case_dir / "export",
            env=env,
            size=size,
            channels_last_4d=_case_uses_channels_last_4d(op_name, size),
        )
    except subprocess.CalledProcessError as error:
        output = (error.stdout or "") + (error.stderr or "")
        record["status"] = "export_failed"
        record["error"] = str(error)
        record["log_tail"] = trim_output(output)
        return record
    record["status"] = "ok"
    record["pte"] = str(pte_path)
    record["pte_bytes"] = pte_path.stat().st_size
    record.update(parse_export_output(export_output))
    host_outputs, inputs, quantized_outputs, reference_name = _delegated_reference_outputs(
        op_name,
        size,
    )
    record["input_elements"] = int(sum(int(tensor.numel()) for tensor in inputs))
    record["reference"] = reference_name
    quantized_metrics = _reference_delta(host_outputs, quantized_outputs)
    record["quantized_vs_reference_max_abs_error"] = quantized_metrics["max_abs_error"]
    record["quantized_vs_reference_rmse"] = quantized_metrics["rmse"]
    _save_tensor_sequence(inputs, case_dir / "inputs", "input")
    _save_tensor_sequence(host_outputs, case_dir, "reference")
    _save_tensor_sequence(quantized_outputs, case_dir, "quantized")
    capture_dir = case_dir / "captures"
    try:
        remote_metrics, local_prefix = _run_remote_capture(
            op_name=op_name,
            size=size,
            pte_path=pte_path,
            runner=runner,
            ssh_target=ssh_target,
            ssh_options=ssh_options,
            remote_dir=remote_dir,
            local_capture_dir=capture_dir,
            inputs=inputs,
            num_executions=num_executions,
        )
    except subprocess.CalledProcessError as error:
        output = (error.stdout or "") + (error.stderr or "")
        record["status"] = "run_failed"
        record["error"] = str(error)
        record["log_tail"] = trim_output(output)
        return record
    record.update(remote_metrics)
    record.update(_compare_outputs(host_outputs, local_prefix))
    return record


def _reference_delta(lhs_outputs, rhs_outputs) -> dict[str, float]:
    lhs_tensors = _tensor_sequence(lhs_outputs)
    rhs_tensors = _tensor_sequence(rhs_outputs)
    max_abs_error = 0.0
    rmse_sum = 0.0
    value_count = 0
    for lhs, rhs in zip(lhs_tensors, rhs_tensors):
        lhs_np = lhs.detach().cpu().numpy().astype(np.float64)
        rhs_np = rhs.detach().cpu().numpy().astype(np.float64)
        diff = lhs_np - rhs_np
        max_abs_error = max(max_abs_error, float(np.max(np.abs(diff))))
        rmse_sum += float(np.sum(diff * diff))
        value_count += int(diff.size)
    return {
        "max_abs_error": max_abs_error,
        "rmse": (rmse_sum / max(value_count, 1)) ** 0.5,
    }


def _delegated_reference_outputs(
    op_name: str,
    size: int,
) -> tuple[object, tuple[torch.Tensor, ...], object, str]:
    channels_last_4d = _case_uses_channels_last_4d(op_name, size)
    module, example_inputs = build_case(op_name, size)
    module, prepared_inputs = arm_aot_compiler.prepare_model_and_inputs_for_export(
        module.eval(),
        execution_inputs(example_inputs),
        channels_last_4d,
    )
    inputs = execution_inputs(prepared_inputs)
    exported = torch_export.export(module, inputs, strict=True)
    exported_module = strip_export_guards(exported.module())
    compile_spec = arm_aot_compiler.get_compile_spec(
        REFERENCE_TARGET,
        system_config=DEFAULT_SYSTEM_CONFIG,
        memory_mode=DEFAULT_MEMORY_MODE,
        quantize=True,
        config="Arm/vela.ini",
        extra_compiler_flags=list(DEFAULT_COMPILER_FLAGS),
    )
    quantized = arm_aot_compiler.quantize(
        exported_module,
        str(Path(__file__).with_name("operator_microbench_model.py")),
        compile_spec,
        inputs,
        None,
        None,
    )
    class Args:
        pass

    args = Args()
    args.target = REFERENCE_TARGET
    args.intermediates = None
    args.system_config = DEFAULT_SYSTEM_CONFIG
    args.memory_mode = DEFAULT_MEMORY_MODE
    args.quantize = True
    args.config = "Arm/vela.ini"
    args.enable_debug_mode = None
    args.direct_drive = False
    args.extra_compiler_flag = list(DEFAULT_COMPILER_FLAGS)
    args.model_name = str(Path(__file__).with_name("operator_microbench_model.py"))
    args.evaluate = None
    args.evaluate_config = None
    args.strict_export = True
    args.channels_last_4d = channels_last_4d
    _, edge = arm_aot_compiler.to_edge_TOSA_delegate(
        exported,
        args,
        exported_module,
        inputs,
    )
    delegated = edge.exported_program().module()
    with torch.no_grad():
        quantized_outputs = quantized(*inputs)
    try:
        with torch.no_grad(), TosaReferenceModelDispatch():
            delegated_outputs = delegated(*inputs)
        reference_name = "delegated_tosa_reference"
    except RuntimeError as error:
        if "never ran TOSABackend delegate" not in str(error):
            raise
        delegated_outputs = quantized_outputs
        reference_name = "host_quantized_model"
    return delegated_outputs, inputs, quantized_outputs, reference_name


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ops", nargs="+", default=list(DEFAULT_SWEEP_OPS))
    parser.add_argument("--sizes", nargs="+", type=int, default=list(DEFAULT_SIZES))
    parser.add_argument("--output", default="/tmp/imx93-operator-sweep")
    parser.add_argument("--summary", default="/tmp/imx93-operator-sweep/summary.json")
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--ssh_target", required=True)
    parser.add_argument("--ssh_option", action="append", default=[])
    parser.add_argument("--remote_dir", default="/tmp/imx93-fastpath-bench")
    parser.add_argument("--num_executions", type=int, default=10)
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for op_name in args.ops:
        for size in args.sizes:
            record = _measure_case(
                op_name=op_name,
                size=size,
                output_dir=output_dir,
                runner=args.runner,
                ssh_target=args.ssh_target,
                ssh_options=args.ssh_option,
                remote_dir=args.remote_dir,
                num_executions=args.num_executions,
            )
            records.append(record)
            print(f"{op_name}:{size} -> {record['status']}")

    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(records, indent=2))
    print(summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
