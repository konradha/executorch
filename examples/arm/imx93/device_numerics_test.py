#!/usr/bin/env python3
"""Direct device numerics test for fastpath models.

Exports models, sends to device, runs, captures output, compares.
No stale references — everything generated fresh in one pass.

Usage:
    python -m examples.arm.imx93.device_numerics_test \
        --models mv2 resnet18 resnet50 ic3 \
        --output /tmp/fastpath-numerics
"""

from __future__ import annotations

import argparse
import ctypes
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import export as torch_export

_EXECUTORCH_DIR = Path(__file__).resolve().parents[3]
_EXECUTORCH_SRC_DIR = _EXECUTORCH_DIR / "src"
if str(_EXECUTORCH_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_EXECUTORCH_SRC_DIR))
if str(_EXECUTORCH_DIR) not in sys.path:
    sys.path.insert(0, str(_EXECUTORCH_DIR))

from examples.arm import aot_arm_compiler
from examples.arm.imx93.export_and_verify import (
    DEFAULT_COMPILER_FLAGS,
    DEFAULT_MEMORY_MODE,
    DEFAULT_SYSTEM_CONFIG,
    strip_export_guards,
)
from examples.models import MODEL_NAME_TO_MODEL
from examples.models.model_factory import EagerModelFactory

SSH_TARGET = "fritz@dev.apparatlabs.com"
SSH_OPTIONS = [
    "-o", "ProxyCommand=cloudflared access ssh --hostname dev.apparatlabs.com",
]
REMOTE_DIR = "/tmp/imx93-numerics-test"
REMOTE_RUNNER = "/tmp/imx93-fastpath-bench/executor_runner"


def ssh_cmd(cmd: str, timeout: int = 60) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["ssh", *SSH_OPTIONS, SSH_TARGET, cmd],
        capture_output=True, text=True, timeout=timeout,
    )


def scp_to(local: str, remote: str) -> None:
    subprocess.run(
        ["scp", *SSH_OPTIONS, local, f"{SSH_TARGET}:{remote}"],
        check=True, capture_output=True, timeout=120,
    )


def scp_from(remote: str, local: str) -> None:
    subprocess.run(
        ["scp", *SSH_OPTIONS, f"{SSH_TARGET}:{remote}", local],
        check=True, capture_output=True, timeout=120,
    )


def tensor_storage_bytes(tensor: torch.Tensor) -> bytes:
    tensor = tensor.detach().cpu()
    return ctypes.string_at(tensor.data_ptr(), tensor.untyped_storage().nbytes())


def export_and_test(model_name: str, output_dir: Path, channels_last: bool) -> dict:
    tag = "clast" if channels_last else "nchw"
    record = {"model": model_name, "layout": tag}

    # 1. Get model and inputs
    model, example_inputs, _, _ = EagerModelFactory.create_model(
        *MODEL_NAME_TO_MODEL[model_name]
    )
    model.eval()

    inputs_raw = (
        (example_inputs[0],)
        if isinstance(example_inputs, tuple)
        else (example_inputs,)
    )
    has_4d = any(t.dim() == 4 for t in inputs_raw)

    if channels_last and has_4d:
        model, prepared = aot_arm_compiler.prepare_model_and_inputs_for_export(
            model, example_inputs, True
        )
        inputs = tuple(
            t
            for t in (prepared if isinstance(prepared, tuple) else (prepared,))
            if isinstance(t, torch.Tensor)
        )
    else:
        inputs = inputs_raw

    # 2. Float reference
    with torch.no_grad():
        float_out = model(*inputs)
        if isinstance(float_out, tuple):
            float_out = float_out[0]
    float_np = float_out.detach().numpy().flatten()
    float_argmax = int(np.argmax(float_np))
    float_top5 = np.argsort(float_np)[-5:][::-1].tolist()

    # 3. Host quantized reference
    exported = torch_export.export(model, inputs, strict=True)
    exported_module = strip_export_guards(exported.module())

    compile_spec = aot_arm_compiler.get_compile_spec(
        "TOSA-1.0+INT",
        system_config=DEFAULT_SYSTEM_CONFIG,
        memory_mode=DEFAULT_MEMORY_MODE,
        quantize=True,
        config="Arm/vela.ini",
        extra_compiler_flags=list(DEFAULT_COMPILER_FLAGS),
    )
    quantized = aot_arm_compiler.quantize(
        exported_module, model_name, compile_spec, inputs, None, None
    )
    with torch.no_grad():
        quant_out = quantized(*inputs)
        if isinstance(quant_out, tuple):
            quant_out = quant_out[0]
    quant_np = quant_out.detach().numpy().flatten()
    quant_argmax = int(np.argmax(quant_np))
    quant_top5 = np.argsort(quant_np)[-5:][::-1].tolist()

    record["float_argmax"] = float_argmax
    record["float_top5"] = float_top5
    record["quant_argmax"] = quant_argmax
    record["quant_top5"] = quant_top5
    record["host_float_vs_quant_rmse"] = float(
        np.sqrt(np.mean((float_np - quant_np) ** 2))
    )

    # 4. Export PTE
    pte_dir = output_dir / f"{model_name}_{tag}"
    pte_dir.mkdir(parents=True, exist_ok=True)

    export_args = [
        sys.executable, "-m", "examples.arm.aot_arm_compiler",
        "-m", model_name, "-t", "ethos-u65-256", "-q", "-d",
        "-o", str(pte_dir),
        "--system_config", DEFAULT_SYSTEM_CONFIG,
        "--memory_mode", DEFAULT_MEMORY_MODE,
    ]
    for flag in DEFAULT_COMPILER_FLAGS:
        export_args.append(f"--extra_compiler_flag={flag}")
    if channels_last:
        export_args.append("--channels_last_4d")

    # Find quantized ops library
    from examples.arm.imx93.export_and_verify import detect_quantized_ops_library
    qops = detect_quantized_ops_library()
    if qops:
        export_args.extend(["-s", qops])

    print(f"  Exporting {model_name} ({tag})...")
    result = subprocess.run(
        export_args, cwd=str(_EXECUTORCH_DIR),
        capture_output=True, text=True, timeout=600,
    )
    if result.returncode != 0:
        record["status"] = "export_failed"
        record["error"] = result.stderr[-500:]
        return record

    pte_files = sorted(pte_dir.glob("*.pte"))
    if not pte_files:
        record["status"] = "export_failed"
        record["error"] = "No PTE file produced"
        return record

    pte_path = max(pte_files, key=lambda p: p.stat().st_size)
    record["pte_bytes"] = pte_path.stat().st_size
    record["pte_path"] = str(pte_path)

    # 5. Send input + PTE to device
    input_bin = pte_dir / "input-0.bin"
    input_bin.write_bytes(tensor_storage_bytes(inputs[0]))

    remote_pte = f"{REMOTE_DIR}/{model_name}_{tag}.pte"
    remote_input = f"{REMOTE_DIR}/{model_name}_{tag}_input.bin"
    remote_out_prefix = f"{REMOTE_DIR}/{model_name}_{tag}_out"

    ssh_cmd(f"mkdir -p {REMOTE_DIR}")
    print(f"  Uploading PTE ({pte_path.stat().st_size // 1024} KB)...")
    scp_to(str(pte_path), remote_pte)
    scp_to(str(input_bin), remote_input)

    # 6. Run on device
    run_cmd = (
        f"rm -f {remote_out_prefix}-*.bin && "
        f"{REMOTE_RUNNER} "
        f"--model_path={remote_pte} "
        f"--inputs={remote_input} "
        f"--num_executions=1 "
        f"--output_file={remote_out_prefix}"
    )
    print(f"  Running on device...")
    result = ssh_cmd(run_cmd, timeout=120)
    combined = (result.stdout or "") + (result.stderr or "")

    if result.returncode != 0:
        record["status"] = "run_failed"
        record["log_tail"] = "\n".join(combined.strip().splitlines()[-10:])
        print(f"  FAILED: {record['log_tail'][-200:]}")
        return record

    # Parse cycle counter
    for line in combined.splitlines():
        if "cycle counter" in line.lower():
            try:
                record["cycles"] = int(line.split(":")[-1].strip())
            except ValueError:
                pass

    # 7. Download output
    ls_result = ssh_cmd(f"ls {remote_out_prefix}-*.bin")
    remote_files = ls_result.stdout.split()
    if not remote_files:
        record["status"] = "no_output"
        return record

    local_out = pte_dir / f"device_out.bin"
    scp_from(remote_files[0].strip(), str(local_out))

    # 8. Compare
    device_raw = local_out.read_bytes()
    n_elements = len(quant_np)
    if len(device_raw) == n_elements * 4:
        device_np = np.frombuffer(device_raw, dtype=np.float32)
    elif len(device_raw) == n_elements:
        device_np = np.frombuffer(device_raw, dtype=np.int8).astype(np.float64)
        quant_np = quant_np  # keep float for comparison
    else:
        record["status"] = "size_mismatch"
        record["device_bytes"] = len(device_raw)
        record["expected_bytes"] = n_elements * 4
        return record

    device_argmax = int(np.argmax(device_np))
    device_top5 = np.argsort(device_np.flatten())[-5:][::-1].tolist()

    rmse_vs_quant = float(np.sqrt(np.mean((device_np - quant_np) ** 2)))
    rmse_vs_float = float(np.sqrt(np.mean((device_np - float_np) ** 2)))
    cos_vs_quant = float(
        np.dot(device_np.flatten(), quant_np.flatten())
        / (np.linalg.norm(device_np) * np.linalg.norm(quant_np) + 1e-8)
    )

    record["status"] = "ok"
    record["device_argmax"] = device_argmax
    record["device_top5"] = device_top5
    record["top1_match_float"] = int(device_argmax == float_argmax)
    record["top1_match_quant"] = int(device_argmax == quant_argmax)
    record["top5_match_float"] = int(device_argmax in float_top5)
    record["top5_match_quant"] = int(device_argmax in quant_top5)
    record["rmse_vs_quant"] = rmse_vs_quant
    record["rmse_vs_float"] = rmse_vs_float
    record["cosine_vs_quant"] = cos_vs_quant
    record["device_output_range"] = [float(device_np.min()), float(device_np.max())]

    status_str = "OK" if record["top1_match_quant"] else "MISMATCH"
    print(
        f"  {status_str}: device_argmax={device_argmax} "
        f"quant_argmax={quant_argmax} float_argmax={float_argmax} "
        f"rmse_q={rmse_vs_quant:.4f} cos_q={cos_vs_quant:.4f}"
    )
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["mv2", "resnet18", "resnet50", "ic3"])
    parser.add_argument("--output", type=Path, default=Path("/tmp/fastpath-numerics"))
    parser.add_argument("--channels-last", action="store_true", default=True)
    parser.add_argument("--no-channels-last", dest="channels_last", action="store_false")
    parser.add_argument("--both-layouts", action="store_true",
                        help="Test both NCHW and channels_last for each model")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    records = []

    layouts = [True, False] if args.both_layouts else [args.channels_last]

    for model_name in args.models:
        for cl in layouts:
            tag = "clast" if cl else "nchw"
            print(f"\n{'='*60}")
            print(f"Model: {model_name}, layout: {tag}")
            print(f"{'='*60}")
            try:
                record = export_and_test(model_name, args.output, cl)
            except Exception as e:
                record = {"model": model_name, "layout": tag, "status": "error", "error": str(e)}
                print(f"  ERROR: {e}")
            records.append(record)

    summary_path = args.output / "summary.json"
    summary_path.write_text(json.dumps(records, indent=2))
    print(f"\nSummary: {summary_path}")

    # Print table
    print(f"\n{'Model':<12} {'Layout':<6} {'Status':<8} {'Dev':<6} {'Quant':<6} {'Float':<6} {'RMSE_q':<8} {'Cos_q':<8} {'Top1_q'}")
    print("-" * 80)
    for r in records:
        print(
            f"{r['model']:<12} {r.get('layout','?'):<6} {r.get('status','?'):<8} "
            f"{r.get('device_argmax','?'):<6} {r.get('quant_argmax','?'):<6} "
            f"{r.get('float_argmax','?'):<6} {r.get('rmse_vs_quant','?'):<8} "
            f"{r.get('cosine_vs_quant','?'):<8} {r.get('top1_match_quant','?')}"
        )


if __name__ == "__main__":
    raise SystemExit(main())
