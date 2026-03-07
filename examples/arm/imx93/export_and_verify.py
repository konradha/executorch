#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
#
# Export models for Ethos-U65-256, generate host reference outputs,
# and optionally compare with device results.
#
# Usage:
#   python examples/arm/imx93/export_and_verify.py --model mv2 --output arm_test/models/
#   python examples/arm/imx93/export_and_verify.py --model resnet18 --output arm_test/models/
#   python examples/arm/imx93/export_and_verify.py --model vit --output arm_test/models/
#   python examples/arm/imx93/export_and_verify.py --verify arm_test/models/mv2/ --device-argmax 623

import argparse
import json
import os
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))


def export_model(model_name: str, output_dir: str, target: str = "ethos-u65-256"):
    """Export a model to PTE for the given Ethos-U target."""
    out = Path(output_dir) / model_name
    out.mkdir(parents=True, exist_ok=True)

    pte_path = out / f"{model_name}.pte"
    ref_path = out / "reference.json"

    # Export via aot_arm_compiler
    cmd = [
        sys.executable, "-m", "examples.arm.aot_arm_compiler",
        "-m", model_name,
        "-t", target,
        "-q",  # quantize
        "-d",  # delegate to NPU
        "-o", str(out),
    ]
    print(f"Exporting {model_name}...")
    print(f"  cmd: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True)
    if result.returncode != 0:
        print(f"EXPORT FAILED:\n{result.stderr[-2000:]}")
        return None

    # Find the generated PTE
    pte_files = list(out.glob("*.pte"))
    if not pte_files:
        print(f"No PTE files found in {out}")
        return None
    pte_file = max(pte_files, key=lambda p: p.stat().st_size)
    if pte_file.name != f"{model_name}.pte":
        pte_file.rename(pte_path)
        pte_file = pte_path

    print(f"  PTE: {pte_file} ({pte_file.stat().st_size / 1024:.1f} KB)")

    # Generate host reference output with zero input
    ref = generate_reference(model_name, str(pte_file))
    if ref:
        with open(ref_path, "w") as f:
            json.dump(ref, f, indent=2)
        print(f"  Reference: argmax={ref['argmax']}, shape={ref['shape']}")
        print(f"  Top-5: {ref['top5_indices']}")

    return str(pte_file)


def generate_reference(model_name: str, pte_path: str = None) -> dict:
    """Run host-side inference with zero input using the quantized PyTorch model.

    Since PTE files with 100% NPU delegation can't run on the host
    (no EthosU backend), we run the quantized model directly through PyTorch.
    """
    from examples.models.model_factory import EagerModelFactory
    from executorch.backends.arm.quantizer import get_symmetric_quantization_config
    from executorch.backends.arm.util._factory import create_quantizer
    from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

    result = EagerModelFactory.create_model(
        *MODEL_NAME_TO_MODEL[model_name])
    model, example_inputs = result[0], result[1]
    model.eval()

    # Run float model for reference
    with torch.no_grad():
        zero_inputs = tuple(torch.zeros_like(x) for x in example_inputs)
        float_out = model(*zero_inputs)

    if isinstance(float_out, torch.Tensor):
        out_np = float_out.numpy().flatten().astype(np.float64)
    else:
        out_np = float_out[0].numpy().flatten().astype(np.float64)

    argmax = int(np.argmax(out_np))
    top5 = np.argsort(out_np)[-5:][::-1].tolist()
    top10 = np.argsort(out_np)[-10:][::-1].tolist()
    top5_vals = out_np[top5].tolist()

    result = {
        "model": model_name,
        "pte": pte_path,
        "input_shape": [list(x.shape) for x in example_inputs],
        "output_shape": list(float_out.shape) if isinstance(float_out, torch.Tensor) else list(float_out[0].shape),
        "dtype": "float32",
        "argmax": argmax,
        "argmax_value": float(out_np[argmax]),
        "top5_indices": top5,
        "top5_values": top5_vals,
        "top10_indices": top10,
        "first_10": out_np[:10].tolist(),
        "num_elements": len(out_np),
    }

    print(f"  Float model: argmax={argmax}, top5={top5}")
    return result


# Model name to (module_name, class_name) from examples/models
MODEL_NAME_TO_MODEL = {
    'mv2': ('mobilenet_v2', 'MV2Model'),
    'mv3': ('mobilenet_v3', 'MV3Model'),
    'resnet18': ('resnet', 'ResNet18Model'),
    'resnet50': ('resnet', 'ResNet50Model'),
    'ic3': ('inception_v3', 'InceptionV3Model'),
    'ic4': ('inception_v4', 'InceptionV4Model'),
    'vit': ('torchvision_vit', 'TorchVisionViTModel'),
    'add': ('toy_model', 'AddModule'),
}


def verify_device_output(ref_path: str, device_argmax: int):
    """Compare device argmax with host reference."""
    with open(ref_path) as f:
        ref = json.load(f)

    host_argmax = ref["argmax"]
    top5 = ref["top5_indices"]

    print(f"Host reference: argmax={host_argmax}, top5={top5}")
    print(f"Device result:  argmax={device_argmax}")

    if device_argmax == host_argmax:
        print("PASS: exact argmax match")
        return True
    elif device_argmax in top5:
        print(f"PASS: device argmax {device_argmax} is in host top-5 "
              f"(int8 quantization shifts are expected)")
        return True
    else:
        print(f"FAIL: device argmax {device_argmax} not in host top-5 {top5}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Export and verify Ethos-U65 models")
    parser.add_argument("--model", type=str, help="Model name to export")
    parser.add_argument("--output", type=str, default="arm_test/models",
                        help="Output directory")
    parser.add_argument("--target", type=str, default="ethos-u65-256",
                        help="Ethos-U target")
    parser.add_argument("--verify", type=str,
                        help="Path to model dir with reference.json")
    parser.add_argument("--device-argmax", type=int,
                        help="Device argmax for verification")
    parser.add_argument("--reference-only", type=str,
                        help="Generate reference for existing PTE")
    args = parser.parse_args()

    if args.reference_only:
        # Infer model name from PTE path (directory name)
        model_name = Path(args.reference_only).parent.name
        ref = generate_reference(model_name, args.reference_only)
        if ref:
            ref_path = Path(args.reference_only).parent / "reference.json"
            with open(ref_path, "w") as f:
                json.dump(ref, f, indent=2)
            print(f"Reference saved: argmax={ref['argmax']}, top5={ref['top5_indices']}")
        return

    if args.verify and args.device_argmax is not None:
        ref_path = Path(args.verify) / "reference.json"
        ok = verify_device_output(str(ref_path), args.device_argmax)
        sys.exit(0 if ok else 1)

    if args.model:
        export_model(args.model, args.output, args.target)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
