#!/usr/bin/env python3
"""Comprehensive operator sweep for i.MX93 Ethos-U65.

Tests every NPU-supported op at multiple sizes. Reports delegation,
execution, and numerical accuracy for each (op, size) pair.

Usage:
    python -m examples.arm.imx93.op_sweep --output /tmp/op-sweep
    python -m examples.arm.imx93.op_sweep --ops add relu conv2d --sizes 8 16
"""

from __future__ import annotations

import argparse
import ctypes
import json
import subprocess
import sys
import traceback
from pathlib import Path

import numpy as np
import torch
from torch import export as torch_export

_EXECUTORCH_DIR = Path(__file__).resolve().parents[3]
if str(_EXECUTORCH_DIR / "src") not in sys.path:
    sys.path.insert(0, str(_EXECUTORCH_DIR / "src"))
if str(_EXECUTORCH_DIR) not in sys.path:
    sys.path.insert(0, str(_EXECUTORCH_DIR))

from backends.arm.scripts import aot_arm_compiler
from examples.arm.imx93.export_and_verify import (
    DEFAULT_COMPILER_FLAGS, DEFAULT_MEMORY_MODE, DEFAULT_SYSTEM_CONFIG,
    detect_quantized_ops_library, strip_export_guards,
)
from examples.arm.imx93.device_numerics_test import (
    ssh_cmd, scp_to, scp_from, REMOTE_DIR, REMOTE_RUNNER, SSH_OPTIONS, SSH_TARGET,
)


def _pat(shape, lo=-1.0, hi=1.0):
    n = 1
    for s in shape:
        n *= s
    return torch.linspace(lo, hi, n).reshape(shape)


# ── Op definitions ──────────────────────────────────────────────────────
# Each entry: (module_factory, input_factory, needs_quantizable_wrapper)
# module_factory(size) -> nn.Module
# input_factory(size) -> tuple[Tensor, ...]

def _unary_module(fn):
    class M(torch.nn.Module):
        def forward(self, x):
            return fn(x)
    return lambda size: M()

def _binary_module(fn):
    class M(torch.nn.Module):
        def forward(self, a, b):
            return fn(a, b)
    return lambda size: M()

def _4d_input(size, channels=8, lo=0.0, hi=1.0):
    return (_pat((1, channels, size, size), lo, hi),)

def _4d_pair(size, channels=4, lo=-1.0, hi=1.0):
    return (
        _pat((1, channels, size, size), lo, 0.0),
        _pat((1, channels, size, size), 0.0, hi),
    )

def _2d_input(size, lo=-1.0, hi=1.0):
    return (_pat((size, size), lo, hi),)

def _make_conv2d(size):
    m = torch.nn.Sequential(
        torch.nn.Conv2d(8, 16, 3, padding=1, bias=True),
        torch.nn.BatchNorm2d(16),
    )
    m.eval()
    return m

def _make_conv_transpose2d(size):
    m = torch.nn.Sequential(
        torch.nn.ConvTranspose2d(8, 16, 3, padding=1, bias=True),
        torch.nn.BatchNorm2d(16),
    )
    m.eval()
    return m

def _make_linear(size):
    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(size, size)
        def forward(self, x):
            return self.fc(x)
    return M()

def _make_pool(pool_cls):
    def factory(size):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.pool = pool_cls(3, stride=2, padding=1)
            def forward(self, x):
                return self.pool(x)
        return M()
    return factory

def _make_adaptive_avg_pool(size):
    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.pool = torch.nn.AdaptiveAvgPool2d(1)
        def forward(self, x):
            return self.pool(x)
    return M()

def _make_depthwise(size):
    m = torch.nn.Conv2d(8, 8, 3, padding=1, groups=8, bias=True)
    m.eval()
    return m

class MeanDimModule(torch.nn.Module):
    def forward(self, x):
        return x.mean(dim=[2, 3], keepdim=True)

class ClampModule(torch.nn.Module):
    def forward(self, x):
        return torch.clamp(x, -0.5, 0.5)

class HardtanhModule(torch.nn.Module):
    def forward(self, x):
        return torch.nn.functional.hardtanh(x)

OP_REGISTRY = {
    # arithmetic
    "add": (_binary_module(torch.add), _4d_pair),
    "sub": (_binary_module(torch.sub), _4d_pair),
    "mul": (_binary_module(torch.mul), _4d_pair),
    "abs": (_unary_module(torch.abs), lambda s: _4d_input(s, lo=-2, hi=2)),
    "neg": (_unary_module(torch.neg), lambda s: _4d_input(s, lo=-1, hi=1)),
    "maximum": (_binary_module(torch.maximum), _4d_pair),
    "minimum": (_binary_module(torch.minimum), _4d_pair),

    # activation
    "relu": (_unary_module(torch.relu), lambda s: _4d_input(s, lo=-2, hi=2)),
    "sigmoid": (_unary_module(torch.sigmoid), lambda s: _4d_input(s, lo=-6, hi=6)),
    "tanh": (_unary_module(torch.tanh), lambda s: _4d_input(s, lo=-3, hi=3)),
    "hardtanh": (lambda s: HardtanhModule(), lambda s: _4d_input(s, lo=-2, hi=2)),
    "hardsigmoid": (_unary_module(torch.nn.functional.hardsigmoid), lambda s: _4d_input(s, lo=-4, hi=4)),
    "hardswish": (_unary_module(torch.nn.functional.hardswish), lambda s: _4d_input(s, lo=-4, hi=4)),
    "elu": (_unary_module(lambda x: torch.nn.functional.elu(x)), lambda s: _4d_input(s, lo=-3, hi=3)),
    "silu": (_unary_module(torch.nn.functional.silu), lambda s: _4d_input(s, lo=-4, hi=4)),

    # convolution
    "conv2d": (_make_conv2d, lambda s: _4d_input(s)),
    "depthwise_conv2d": (_make_depthwise, lambda s: _4d_input(s)),
    "conv_transpose2d": (_make_conv_transpose2d, lambda s: _4d_input(s)),

    # pooling
    "avg_pool2d": (_make_pool(torch.nn.AvgPool2d), lambda s: _4d_input(s)),
    "max_pool2d": (_make_pool(torch.nn.MaxPool2d), lambda s: _4d_input(s)),
    "adaptive_avg_pool2d": (_make_adaptive_avg_pool, lambda s: _4d_input(s)),

    # linear
    "linear": (_make_linear, _2d_input),

    # reduction
    "mean_dim": (lambda s: MeanDimModule(), lambda s: _4d_input(s)),
    "clamp": (lambda s: ClampModule(), lambda s: _4d_input(s, lo=-2, hi=2)),

    # math (unary, use TABLE on NPU)
    "exp": (_unary_module(torch.exp), lambda s: _4d_input(s, lo=-2, hi=2)),
    "log": (_unary_module(torch.log), lambda s: _4d_input(s, lo=0.1, hi=5)),
    "ceil": (_unary_module(torch.ceil), lambda s: _4d_input(s, lo=-3, hi=3)),
    "floor": (_unary_module(torch.floor), lambda s: _4d_input(s, lo=-3, hi=3)),
    "rsqrt": (_unary_module(torch.rsqrt), lambda s: _4d_input(s, lo=0.1, hi=5)),

    # comparison (return bool, may not delegate)
    "eq": (_binary_module(torch.eq), _4d_pair),
    "gt": (_binary_module(torch.gt), _4d_pair),
    "le": (_binary_module(torch.le), _4d_pair),
}


def tensor_bytes(t):
    t = t.detach().cpu()
    return ctypes.string_at(t.data_ptr(), t.untyped_storage().nbytes())


def test_op(op_name, size, output_dir):
    record = {"op": op_name, "size": size}

    if op_name not in OP_REGISTRY:
        record["status"] = "not_implemented"
        return record

    module_factory, input_factory = OP_REGISTRY[op_name]

    try:
        model = module_factory(size)
        if not isinstance(model, torch.nn.Module):
            model = model()
        model.eval()
        inputs = input_factory(size)
    except Exception as e:
        record["status"] = "build_failed"
        record["error"] = str(e)
        return record

    # Float reference
    try:
        with torch.no_grad():
            float_out = model(*inputs)
        if isinstance(float_out, tuple):
            float_out = float_out[0]
        float_np = float_out.detach().numpy().flatten().astype(np.float64)
        record["float_dtype"] = str(float_out.dtype)
    except Exception as e:
        record["status"] = "float_failed"
        record["error"] = str(e)
        return record

    # Export + quantize
    try:
        exported = torch_export.export(model, inputs, strict=True)
        gm = strip_export_guards(exported.module())

        compile_spec = aot_arm_compiler.get_compile_spec(
            "ethos-u65-256",
            system_config=DEFAULT_SYSTEM_CONFIG,
            memory_mode=DEFAULT_MEMORY_MODE,
            quantize=True,
            config="Arm/vela.ini",
            extra_compiler_flags=list(DEFAULT_COMPILER_FLAGS),
        )
        quantized = aot_arm_compiler.quantize(gm, op_name, compile_spec, inputs)
        with torch.no_grad():
            q_out = quantized(*inputs)
        if isinstance(q_out, tuple):
            q_out = q_out[0]
        quant_np = q_out.detach().numpy().flatten().astype(np.float64)
    except Exception as e:
        record["status"] = "quantize_failed"
        record["error"] = str(e)[:300]
        return record

    # Export PTE
    op_dir = output_dir / f"{op_name}_{size}"
    op_dir.mkdir(parents=True, exist_ok=True)
    try:
        from executorch.exir import to_edge_transform_and_lower, EdgeCompileConfig
        from executorch.backends.arm.ethosu import EthosUPartitioner

        # Load quantized ops library if available
        qops = detect_quantized_ops_library()
        if qops:
            import ctypes as _ctypes
            try:
                _ctypes.cdll.LoadLibrary(qops)
            except OSError:
                pass

        edge = to_edge_transform_and_lower(
            torch_export.export(quantized, inputs, strict=True),
            partitioner=[EthosUPartitioner(compile_spec)],
            compile_config=EdgeCompileConfig(_check_ir_validity=False),
        )

        # Count delegated vs CPU ops
        n_delegate = 0
        cpu_ops = []
        for node in edge.exported_program().graph_module.graph.nodes:
            if node.op == "get_attr" and "lowered_module" in node.name:
                n_delegate += 1
            elif node.op == "call_function":
                name = str(node.target)
                if "getitem" not in name and "quantize" not in name and "dequantize" not in name:
                    cpu_ops.append(name)

        record["n_delegate_segments"] = n_delegate
        record["cpu_ops"] = cpu_ops
        record["delegated"] = n_delegate > 0

        if n_delegate == 0:
            record["status"] = "not_delegated"
            return record

        # Serialize
        et = edge.to_executorch()
        pte_path = op_dir / f"{op_name}_{size}.pte"
        with open(pte_path, "wb") as f:
            f.write(et.buffer)
        record["pte_bytes"] = pte_path.stat().st_size

    except Exception as e:
        record["status"] = "export_failed"
        record["error"] = str(e)[:300]
        return record

    # Run on device
    try:
        remote_pte = f"{REMOTE_DIR}/sweep_{op_name}_{size}.pte"
        remote_out = f"{REMOTE_DIR}/sweep_{op_name}_{size}_out"
        ssh_cmd(f"mkdir -p {REMOTE_DIR}")
        scp_to(str(pte_path), remote_pte)

        # Send inputs
        remote_inputs = []
        for i, inp in enumerate(inputs):
            local_bin = op_dir / f"input_{i}.bin"
            local_bin.write_bytes(tensor_bytes(inp))
            remote_in = f"{REMOTE_DIR}/sweep_{op_name}_{size}_in{i}.bin"
            scp_to(str(local_bin), remote_in)
            remote_inputs.append(remote_in)

        run_cmd = (
            f"rm -f {remote_out}-*.bin && "
            f"{REMOTE_RUNNER} "
            f"--model_path={remote_pte} "
            f"--inputs={','.join(remote_inputs)} "
            f"--num_executions=1 "
            f"--output_file={remote_out}"
        )
        result = ssh_cmd(run_cmd, timeout=60)
        combined = (result.stdout or "") + (result.stderr or "")

        if result.returncode != 0:
            record["status"] = "device_failed"
            record["log_tail"] = "\n".join(combined.strip().splitlines()[-5:])
            return record

        # Parse cycles
        for line in combined.splitlines():
            if "cycle counter" in line.lower():
                try:
                    record["cycles"] = int(line.split(":")[-1].strip())
                except ValueError:
                    pass

        # Download output
        ls_result = ssh_cmd(f"ls {remote_out}-*.bin 2>/dev/null")
        remote_files = ls_result.stdout.split()
        if not remote_files:
            record["status"] = "no_output"
            return record

        local_out = op_dir / "device_out.bin"
        scp_from(remote_files[0].strip(), str(local_out))

        # Compare
        device_raw = local_out.read_bytes()
        if len(device_raw) == 0:
            record["status"] = "empty_output"
            return record

        # Try to interpret as same dtype as quant output
        if len(device_raw) == len(quant_np) * 4:
            device_np = np.frombuffer(device_raw, dtype=np.float32).astype(np.float64)
        elif len(device_raw) == len(quant_np):
            device_np = np.frombuffer(device_raw, dtype=np.int8).astype(np.float64)
        else:
            record["status"] = "size_mismatch"
            record["device_bytes"] = len(device_raw)
            record["expected_bytes"] = len(quant_np) * 4
            return record

        rmse = float(np.sqrt(np.mean((device_np - quant_np) ** 2)))
        cos = float(
            np.dot(device_np.flatten(), quant_np.flatten())
            / (np.linalg.norm(device_np) * np.linalg.norm(quant_np) + 1e-12)
        )
        record["rmse"] = rmse
        record["cosine"] = cos
        record["correct"] = cos > 0.9
        record["status"] = "ok" if cos > 0.9 else "numerics_wrong"

    except Exception as e:
        record["status"] = "device_error"
        record["error"] = str(e)[:300]

    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ops", nargs="+", default=list(OP_REGISTRY.keys()))
    parser.add_argument("--sizes", nargs="+", type=int, default=[8, 16, 32])
    parser.add_argument("--output", type=Path, default=Path("/tmp/op-sweep"))
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    records = []

    total = len(args.ops) * len(args.sizes)
    done = 0

    for op_name in args.ops:
        for size in args.sizes:
            done += 1
            sys.stdout.write(f"\r[{done}/{total}] {op_name} size={size}...")
            sys.stdout.flush()
            try:
                record = test_op(op_name, size, args.output)
            except Exception as e:
                record = {"op": op_name, "size": size, "status": "crash", "error": traceback.format_exc()[-300:]}
            records.append(record)

            status = record.get("status", "?")
            cos = record.get("cosine", "")
            cos_str = f" cos={cos:.3f}" if isinstance(cos, float) else ""
            sys.stdout.write(f" {status}{cos_str}\n")

    summary_path = args.output / "sweep_summary.json"
    summary_path.write_text(json.dumps(records, indent=2))

    # Print matrix
    print(f"\n{'Op':<22} " + " ".join(f"{'s=' + str(s):<14}" for s in args.sizes))
    print("-" * (22 + 15 * len(args.sizes)))
    for op_name in args.ops:
        row = f"{op_name:<22} "
        for size in args.sizes:
            r = next((r for r in records if r["op"] == op_name and r["size"] == size), None)
            if r is None:
                row += f"{'?':<14} "
            else:
                s = r.get("status", "?")
                if s == "ok":
                    cos_val = r.get("cosine", 0)
                    cell = f"OK {cos_val:.3f}"
                    row += f"{cell:<14} "
                else:
                    row += f"{s[:13]:<14} "
        print(row)

    print(f"\nSummary: {summary_path}")


if __name__ == "__main__":
    raise SystemExit(main())
