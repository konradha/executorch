#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
#
# Synthetic large model (~100M+ params) using only Ethos-U65 compatible ops.
# All conv2d + relu + avg_pool2d — fully delegatable to NPU.
#
# Usage:
#   python -m examples.arm.imx93.large_model  # export PTE
#   python -m examples.arm.imx93.large_model --check  # just count params

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))


class LargeConvNet(nn.Module):
    """~105M parameter ConvNet using only Ethos-U65 friendly ops.

    Architecture: repeated conv blocks with increasing then decreasing channels.
    Input: [1, 3, 64, 64] — small spatial dims to keep activation memory low.
    Output: [1, 1000] — ImageNet-style classifier.
    """

    def __init__(self):
        super().__init__()
        layers = []

        # Stage 1: 3 -> 128
        for i in range(2):
            in_c = 3 if i == 0 else 128
            layers.append(nn.Conv2d(in_c, 128, 3, padding=1, bias=False))
            layers.append(nn.ReLU())

        # Stage 2: 128 -> 512 (~6M)
        for i in range(4):
            in_c = 128 if i == 0 else 512
            layers.append(nn.Conv2d(in_c, 512, 3, padding=1, bias=False))
            layers.append(nn.ReLU())
        layers.append(nn.AvgPool2d(2))  # 64->32

        # Stage 3: 512 -> 1024 (~38M)
        for i in range(6):
            in_c = 512 if i == 0 else 1024
            layers.append(nn.Conv2d(in_c, 1024, 3, padding=1, bias=False))
            layers.append(nn.ReLU())
        layers.append(nn.AvgPool2d(2))  # 32->16

        # Stage 4: 1024 -> 1024 (~56M)
        for i in range(4):
            layers.append(nn.Conv2d(1024, 1024, 3, padding=1, bias=False))
            layers.append(nn.ReLU())

        layers.append(nn.AdaptiveAvgPool2d(1))

        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(1024, 1000, bias=False)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def export_large_model(output_dir: str, target: str = "ethos-u65-256"):
    """Export the large model for Ethos-U65 using aot_arm_compiler."""
    import json
    import subprocess
    import numpy as np

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    model = LargeConvNet()
    model.eval()
    params = count_params(model)
    print(f"LargeConvNet: {params/1e6:.1f}M parameters ({params:,})")

    # Save model so aot_arm_compiler can load it
    model_path = out / "large_convnet_model.pt"
    example_input = torch.randn(1, 3, 64, 64)
    torch.save({"model": model.state_dict(), "input_shape": [1, 3, 64, 64]}, model_path)

    # Use aot_arm_compiler subprocess — matches the exact pattern from
    # examples/arm/aot_arm_compiler.py:to_edge_TOSA_delegate()
    cmd = [
        sys.executable, "-c", f"""
import sys; sys.path.insert(0, '.')
import torch
from examples.arm.imx93.large_model import LargeConvNet

from executorch.backends.arm.ethosu import EthosUCompileSpec
from executorch.backends.arm.quantizer import get_symmetric_quantization_config
from executorch.backends.arm.util._factory import create_partitioner, create_quantizer
from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower
from executorch.extension.export_util.utils import save_pte_program
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

model = LargeConvNet().eval()
example_input = (torch.randn(1, 3, 64, 64),)

compile_spec = EthosUCompileSpec('{target}',
    extra_flags=['--verbose-operators', '--verbose-cycle-estimate'])

# Export first, then get GraphModule for quantization
exported_program = torch.export.export(model, example_input, strict=True)
gm = exported_program.module()

# Strip _guards_fn call_module nodes (incompatible with ARM ExportPass)
for node in list(gm.graph.nodes):
    if node.op == 'call_module' and node.target == '_guards_fn':
        gm.graph.erase_node(node)
        if hasattr(gm, '_guards_fn'):
            delattr(gm, '_guards_fn')
gm.graph.lint()
gm.recompile()

# Quantize the GraphModule
quantizer = create_quantizer(compile_spec)
qconfig = get_symmetric_quantization_config(is_per_channel=True)
quantizer.set_global(qconfig)
prepared = prepare_pt2e(gm, quantizer)
prepared(*example_input)
quantized = convert_pt2e(prepared)

# Re-export quantized model
exported = torch.export.export(quantized, example_input, strict=True)

# Partition and lower
partitioner = create_partitioner(compile_spec)
edge = to_edge_transform_and_lower(
    exported,
    partitioner=[partitioner],
    compile_config=EdgeCompileConfig(_check_ir_validity=False),
)

pte_path = '{out}/large_convnet_arm_delegate_{target}.pte'
save_pte_program(edge.to_executorch(), pte_path)
import os
print(f'PTE saved: {{pte_path}} ({{os.path.getsize(pte_path)/1024/1024:.1f}} MB)')
"""
    ]
    print("Exporting (this may take a while for 98M params)...")
    result = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True,
                          timeout=600)
    print(result.stdout[-500:] if result.stdout else "")
    if result.returncode != 0:
        print(f"STDERR: {result.stderr[-2000:]}")
        return None

    pte_path = out / f"large_convnet_arm_delegate_{target}.pte"
    if pte_path.exists():
        print(f"PTE: {pte_path} ({pte_path.stat().st_size / 1024 / 1024:.1f} MB)")
    else:
        print("PTE not created")
        return None

    # Generate reference
    with torch.no_grad():
        zero_input = torch.zeros(1, 3, 64, 64)
        ref_out = model(zero_input).numpy().flatten()

    argmax = int(np.argmax(ref_out))
    top5 = np.argsort(ref_out)[-5:][::-1].tolist()
    ref = {
        "model": "large_convnet",
        "params_millions": round(params / 1e6, 1),
        "input_shape": [1, 3, 64, 64],
        "output_shape": [1, 1000],
        "argmax": argmax,
        "top5_indices": top5,
        "top5_values": ref_out[top5].tolist(),
        "first_10": ref_out[:10].tolist(),
        "num_elements": len(ref_out),
    }
    ref_path = out / "reference.json"
    with open(ref_path, "w") as f:
        json.dump(ref, f, indent=2)
    print(f"Reference: argmax={argmax}, top5={top5}")

    return str(pte_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="Just count params")
    parser.add_argument("--output", default="arm_test/models/large_convnet")
    parser.add_argument("--target", default="ethos-u65-256")
    args = parser.parse_args()

    if args.check:
        m = LargeConvNet()
        print(f"LargeConvNet: {count_params(m)/1e6:.1f}M params")
        print(f"Input: [1, 3, 64, 64]")
        print(f"Output: [1, 1000]")
    else:
        export_large_model(args.output, args.target)
