#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import argparse
import ctypes
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import export as torch_export

_EXECUTORCH_DIR = Path(__file__).resolve().parents[3]
_EXECUTORCH_SRC_DIR = _EXECUTORCH_DIR / "src"
_EXECUTORCH_SRC_DIR_STR = str(_EXECUTORCH_SRC_DIR)
_EXECUTORCH_DIR_STR = str(_EXECUTORCH_DIR)
if _EXECUTORCH_SRC_DIR_STR not in sys.path:
    sys.path.insert(0, _EXECUTORCH_SRC_DIR_STR)
if _EXECUTORCH_DIR_STR not in sys.path:
    sys.path.insert(0, _EXECUTORCH_DIR_STR)

from backends.arm.scripts import aot_arm_compiler
from examples.arm.imx93.export_and_verify import (
    DEFAULT_COMPILER_FLAGS,
    DEFAULT_MEMORY_MODE,
    DEFAULT_SYSTEM_CONFIG,
    strip_export_guards,
)
from examples.arm.imx93.operator_benchmarks import build_ssh_command, parse_runner_output
from examples.models import MODEL_NAME_TO_MODEL
from examples.models.model_factory import EagerModelFactory


REPO_ROOT = _EXECUTORCH_DIR
DEFAULT_REFERENCE_ROOT = REPO_ROOT.parent / "executorch" / "arm_test" / "models"
DEFAULT_MODELS = (
    "mv3",
    "mv2",
    "resnet18",
    "resnet50",
    "ic3",
    "large_convnet",
)
MODEL_PARAMS_MILLIONS = {
    "mv3": 2.5,
    "mv2": 3.5,
    "resnet18": 11.7,
    "resnet50": 25.6,
    "ic3": 27.2,
    "large_convnet": 105.0,
}
TRANSFER_RETRIES = 3
TRANSFER_TIMEOUT_SECONDS = 45
_STAGED_REMOTE_RUNNERS: set[tuple[str, tuple[str, ...], str]] = set()
REFERENCE_TARGET = "TOSA-1.0+INT"


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


def _scp_with_options(source: str, destination: str, ssh_options: list[str]) -> None:
    command = ["scp"]
    for option in ssh_options:
        command.extend(["-o", option])
    command.extend([source, destination])
    _run_with_retries(command)


def _ensure_remote_runner(
    runner: Path,
    ssh_target: str,
    ssh_options: list[str],
    remote_dir: str,
) -> str:
    remote_runner = f"{remote_dir.rstrip('/')}/{runner.name}"
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


def _tensor_sequence(output) -> list[torch.Tensor]:
    if isinstance(output, tuple):
        return list(output)
    if isinstance(output, list):
        return output
    return [output]


def _clone_input_structure(value):
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, tuple):
        return tuple(_clone_input_structure(item) for item in value)
    if isinstance(value, list):
        return [_clone_input_structure(item) for item in value]
    return value


def _zero_like_structure(value):
    if isinstance(value, torch.Tensor):
        return torch.zeros_like(value, memory_format=torch.preserve_format)
    if isinstance(value, tuple):
        return tuple(_zero_like_structure(item) for item in value)
    if isinstance(value, list):
        return [_zero_like_structure(item) for item in value]
    return value


def _input_tensor_sequence(example_inputs) -> tuple[torch.Tensor, ...]:
    if isinstance(example_inputs, tuple):
        return tuple(item for item in example_inputs if isinstance(item, torch.Tensor))
    if isinstance(example_inputs, list):
        return tuple(item for item in example_inputs if isinstance(item, torch.Tensor))
    if isinstance(example_inputs, torch.Tensor):
        return (example_inputs,)
    raise TypeError(f"Unsupported example input type: {type(example_inputs)!r}")


def _comparison_tensor(tensor: torch.Tensor) -> torch.Tensor:
    tensor = tensor.detach().cpu()
    if tensor.dim() == 4 and tensor.is_contiguous(memory_format=torch.channels_last):
        return tensor
    if tensor.is_contiguous():
        return tensor
    return tensor.contiguous()


def _tensor_storage_bytes(tensor: torch.Tensor) -> bytes:
    tensor = tensor.detach().cpu()
    return ctypes.string_at(tensor.data_ptr(), tensor.untyped_storage().nbytes())


def _load_device_tensor(local_file: Path, host_tensor: torch.Tensor) -> np.ndarray:
    host_tensor = _comparison_tensor(host_tensor)
    host = host_tensor.numpy()
    element_count = int(host.size)
    file_size = local_file.stat().st_size
    if element_count == 0 or file_size % max(element_count, 1) != 0:
        raise ValueError(
            f"Unexpected capture size for {local_file}: {file_size} bytes for {element_count} elements"
        )
    bytes_per_element = file_size // max(element_count, 1)
    if host.dtype.kind in ("i", "u"):
        device_dtype = {
            1: np.int8 if host.dtype.kind == "i" else np.uint8,
            2: np.int16 if host.dtype.kind == "i" else np.uint16,
            4: np.int32 if host.dtype.kind == "i" else np.uint32,
            8: np.int64 if host.dtype.kind == "i" else np.uint64,
        }.get(bytes_per_element)
    elif host.dtype.kind == "f":
        device_dtype = {2: np.float16, 4: np.float32, 8: np.float64}.get(bytes_per_element)
    elif host.dtype.kind == "b":
        device_dtype = np.bool_ if bytes_per_element == 1 else None
    else:
        device_dtype = None
    if device_dtype is None:
        raise ValueError(f"Unsupported capture width {bytes_per_element} for {local_file}")
    raw = local_file.read_bytes()
    itemsize = np.dtype(device_dtype).itemsize
    strides = tuple(int(stride) * itemsize for stride in host_tensor.stride())
    return np.ndarray(shape=host.shape, dtype=device_dtype, buffer=raw, strides=strides).copy()


def _model_reference(model_name: str) -> tuple[tuple[torch.Tensor, ...], list[torch.Tensor], str] | None:
    if model_name not in MODEL_NAME_TO_MODEL:
        return None
    model, example_inputs, _, _ = EagerModelFactory.create_model(*MODEL_NAME_TO_MODEL[model_name])
    model = model.eval()
    model, prepared_inputs = aot_arm_compiler.prepare_model_and_inputs_for_export(
        model,
        _clone_input_structure(example_inputs),
        False,
    )
    inputs = _input_tensor_sequence(_clone_input_structure(prepared_inputs))
    exported = torch_export.export(model, inputs, strict=True)
    exported_module = strip_export_guards(exported.module())
    compile_spec = aot_arm_compiler.get_compile_spec(
        REFERENCE_TARGET,
        system_config=DEFAULT_SYSTEM_CONFIG,
        memory_mode=DEFAULT_MEMORY_MODE,
        quantize=True,
        config="Arm/vela.ini",
        extra_compiler_flags=list(DEFAULT_COMPILER_FLAGS),
    )
    quantized = aot_arm_compiler.quantize(exported_module, model_name, compile_spec, inputs)
    with torch.no_grad():
        outputs = _tensor_sequence(quantized(*inputs))
    return inputs, outputs, "host_quantized_model"


def _read_reference(reference_root: Path, model_name: str) -> dict[str, object]:
    reference_path = reference_root / model_name / "reference.json"
    return json.loads(reference_path.read_text())


def _resolve_pte_path(
    reference_root: Path,
    model_name: str,
    reference: dict[str, object],
) -> Path:
    if "pte" in reference:
        return Path(reference["pte"]).expanduser().resolve()
    candidates = sorted((reference_root / model_name).glob("*.pte"))
    if not candidates:
        raise FileNotFoundError(f"No .pte file found for {model_name}")
    return max(candidates, key=lambda candidate: candidate.stat().st_size)


def summarize_device_output(
    output: np.ndarray,
    reference_output: np.ndarray,
) -> dict[str, object]:
    flat = output.astype(np.float64).reshape(-1)
    reference_flat = reference_output.astype(np.float64).reshape(-1)
    top5 = np.argsort(flat)[-5:][::-1].tolist()
    reference_top5 = np.argsort(reference_flat)[-5:][::-1].tolist()
    first_10 = reference_flat[:10]
    first_10_device = flat[: len(first_10)]
    first_10_diff = np.abs(first_10_device - first_10)
    diff = flat - reference_flat
    max_abs_error = float(np.max(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff * diff)))
    device_argmax = int(np.argmax(flat))
    reference_argmax = int(np.argmax(reference_flat))
    return {
        "device_argmax": device_argmax,
        "device_top5": top5,
        "reference_argmax": reference_argmax,
        "reference_top5": reference_top5,
        "top1_match": int(device_argmax == reference_argmax),
        "top5_match": int(device_argmax in reference_top5),
        "max_abs_error": max_abs_error,
        "rmse": rmse,
        "first10_max_abs_error": float(np.max(first_10_diff)),
        "first10_rmse": float(np.sqrt(np.mean(first_10_diff * first_10_diff))),
    }


def _run_model(
    *,
    model_name: str,
    reference: dict[str, object],
    reference_root: Path,
    runner: Path,
    ssh_target: str,
    ssh_options: list[str],
    remote_dir: str,
    local_output_dir: Path,
    num_executions: int,
) -> dict[str, object]:
    record: dict[str, object] = {
        "model": model_name,
        "params_millions": MODEL_PARAMS_MILLIONS.get(model_name),
    }
    reference_bundle = _model_reference(model_name)
    if reference_bundle is None:
        record["status"] = "error"
        record["error"] = f"No host reference pipeline for model {model_name}"
        return record
    inputs, reference_outputs, reference_name = reference_bundle
    pte_path = _resolve_pte_path(reference_root, model_name, reference)
    record["pte"] = str(pte_path)
    record["pte_bytes"] = pte_path.stat().st_size
    record["reference"] = reference_name
    remote_runner = _ensure_remote_runner(runner, ssh_target, ssh_options, remote_dir)
    remote_model = f"{remote_dir.rstrip('/')}/{model_name}.pte"
    remote_prefix = f"{remote_dir.rstrip('/')}/{model_name}-out"
    _scp_with_options(str(pte_path), f"{ssh_target}:{remote_model}", ssh_options)
    remote_inputs = []
    for index, tensor in enumerate(inputs):
        local_input = local_output_dir / f"{model_name}-input-{index}.bin"
        local_input.parent.mkdir(parents=True, exist_ok=True)
        local_input.write_bytes(_tensor_storage_bytes(tensor))
        remote_input = f"{remote_dir.rstrip('/')}/{model_name}-input-{index}.bin"
        _scp_with_options(str(local_input), f"{ssh_target}:{remote_input}", ssh_options)
        remote_inputs.append(remote_input)
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
    try:
        result = _run_with_retries(
            build_ssh_command(ssh_target, remote_command, ssh_options)
        )
    except subprocess.CalledProcessError as error:
        output = (error.stdout or "") + (error.stderr or "")
        record["status"] = "run_failed"
        record["log_tail"] = "\n".join(output.strip().splitlines()[-20:])
        return record
    record["status"] = "ok"
    record.update(parse_runner_output(result.stdout + result.stderr))
    remote_list = _run_with_retries(
        build_ssh_command(
            ssh_target,
            f"ls {shlex.quote(remote_prefix)}-*.bin",
            ssh_options,
        )
    )
    remote_files = remote_list.stdout.split()
    if not remote_files:
        record["status"] = "run_failed"
        record["log_tail"] = "runner did not emit output_file captures"
        return record
    local_output_dir.mkdir(parents=True, exist_ok=True)
    for remote_file in remote_files:
        _scp_with_options(
            f"{ssh_target}:{remote_file}",
            str(local_output_dir / Path(remote_file).name),
            ssh_options,
        )
    device_output = _load_device_tensor(
        local_output_dir / Path(remote_files[0]).name,
        reference_outputs[0],
    )
    record.update(
        summarize_device_output(
            device_output,
            _comparison_tensor(reference_outputs[0]).numpy(),
        )
    )
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--reference_root", type=Path, default=DEFAULT_REFERENCE_ROOT)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--ssh_target", required=True)
    parser.add_argument("--ssh_option", action="append", default=[])
    parser.add_argument("--remote_dir", default="/tmp/imx93-fastpath-bench")
    parser.add_argument("--num_executions", type=int, default=10)
    parser.add_argument("--output", default="/tmp/imx93-model-benchmarks")
    parser.add_argument("--summary", default="/tmp/imx93-model-benchmarks/summary.json")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for model_name in args.models:
        reference = _read_reference(args.reference_root, model_name)
        try:
            record = _run_model(
                model_name=model_name,
                reference=reference,
                reference_root=args.reference_root,
                runner=args.runner,
                ssh_target=args.ssh_target,
                ssh_options=args.ssh_option,
                remote_dir=args.remote_dir,
                local_output_dir=output_dir / model_name,
                num_executions=args.num_executions,
            )
        except Exception as error:
            record = {
                "model": model_name,
                "status": "error",
                "error": str(error),
                "params_millions": MODEL_PARAMS_MILLIONS.get(model_name),
            }
        records.append(record)
        print(f"{model_name} -> {record['status']}")
    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(records, indent=2))
    print(summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
