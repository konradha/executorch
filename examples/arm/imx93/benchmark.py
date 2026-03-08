#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
#
# Benchmark classification models: float32 vs int8 quantized precision.
# Uses real images (ImageNet-style preprocessing) and compares outputs.
#
# Usage:
#   python -m examples.arm.imx93.benchmark --image dog.jpg
#   python -m examples.arm.imx93.benchmark --image dog.jpg --models mv2 resnet18 resnet50

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

# ImageNet class labels (top-level subset for display)
IMAGENET_LABELS_URL = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"

MODELS = {
    "mv2": ("mobilenet_v2", "MV2Model"),
    "mv3": ("mobilenet_v3", "MV3Model"),
    "resnet18": ("resnet", "ResNet18Model"),
    "resnet50": ("resnet", "ResNet50Model"),
    "ic3": ("inception_v3", "InceptionV3Model"),
}

# Standard ImageNet preprocessing
IMAGENET_PREPROCESS = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def load_image(path: str) -> torch.Tensor:
    from PIL import Image
    img = Image.open(path).convert("RGB")
    tensor = IMAGENET_PREPROCESS(img).unsqueeze(0)
    return tensor


def load_labels() -> list:
    try:
        labels_path = REPO_ROOT / "arm_test" / "imagenet_classes.txt"
        if labels_path.exists():
            return labels_path.read_text().strip().split("\n")
        import urllib.request
        data = urllib.request.urlopen(IMAGENET_LABELS_URL).read().decode()
        labels_path.parent.mkdir(parents=True, exist_ok=True)
        labels_path.write_text(data)
        return data.strip().split("\n")
    except Exception:
        return [str(i) for i in range(1000)]


def quantize_model(model, example_input):
    """Quantize using the same pipeline as aot_arm_compiler."""
    from executorch.backends.arm.ethosu import EthosUCompileSpec
    from executorch.backends.arm.quantizer import get_symmetric_quantization_config
    from executorch.backends.arm.util._factory import create_quantizer
    from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

    compile_spec = EthosUCompileSpec("ethos-u65-256")

    exported = torch.export.export(model, example_input, strict=True)
    gm = exported.module()

    # Strip _guards_fn (see pytorch bug: call_module incompatible with ExportPass)
    for node in list(gm.graph.nodes):
        if node.op == "call_module" and node.target == "_guards_fn":
            gm.graph.erase_node(node)
            if hasattr(gm, "_guards_fn"):
                delattr(gm, "_guards_fn")
    gm.graph.lint()
    gm.recompile()

    quantizer = create_quantizer(compile_spec)
    qconfig = get_symmetric_quantization_config(is_per_channel=True)
    quantizer.set_global(qconfig)

    prepared = prepare_pt2e(gm, quantizer)
    prepared(*example_input)
    quantized = convert_pt2e(prepared)
    return quantized


def benchmark_model(model_name, image_tensor, labels):
    """Compare float32 vs int8 outputs for a single model."""
    from examples.models.model_factory import EagerModelFactory

    result = EagerModelFactory.create_model(*MODELS[model_name])
    model = result[0].eval()

    # Float inference
    with torch.no_grad():
        float_out = model(image_tensor)
    if not isinstance(float_out, torch.Tensor):
        float_out = float_out[0]
    float_np = float_out.numpy().flatten()
    float_probs = F.softmax(float_out, dim=1).numpy().flatten()

    # Quantized inference
    quantized = quantize_model(model, (image_tensor,))
    with torch.no_grad():
        quant_out = quantized(image_tensor)
    if not isinstance(quant_out, torch.Tensor):
        quant_out = quant_out[0]
    quant_np = quant_out.numpy().flatten()
    quant_probs = F.softmax(quant_out, dim=1).numpy().flatten()

    # Metrics
    float_argmax = int(np.argmax(float_np))
    quant_argmax = int(np.argmax(quant_np))
    float_top5 = np.argsort(float_np)[-5:][::-1].tolist()
    quant_top5 = np.argsort(quant_np)[-5:][::-1].tolist()

    # Error metrics (on logits)
    abs_err = np.abs(float_np - quant_np)
    max_err = float(np.max(abs_err))
    mean_err = float(np.mean(abs_err))
    # Cosine similarity
    cos_sim = float(np.dot(float_np, quant_np) /
                    (np.linalg.norm(float_np) * np.linalg.norm(quant_np) + 1e-8))
    # Top-5 overlap
    top5_overlap = len(set(float_top5) & set(quant_top5))
    # KL divergence on probabilities
    kl_div = float(np.sum(float_probs * np.log((float_probs + 1e-10) /
                                                (quant_probs + 1e-10))))

    return {
        "model": model_name,
        "float_argmax": float_argmax,
        "float_label": labels[float_argmax] if float_argmax < len(labels) else str(float_argmax),
        "float_top5": float_top5,
        "float_top5_labels": [labels[i] if i < len(labels) else str(i) for i in float_top5],
        "quant_argmax": quant_argmax,
        "quant_label": labels[quant_argmax] if quant_argmax < len(labels) else str(quant_argmax),
        "quant_top5": quant_top5,
        "quant_top5_labels": [labels[i] if i < len(labels) else str(i) for i in quant_top5],
        "argmax_match": float_argmax == quant_argmax,
        "top5_overlap": top5_overlap,
        "cosine_similarity": cos_sim,
        "max_abs_error": max_err,
        "mean_abs_error": mean_err,
        "kl_divergence": kl_div,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--models", nargs="+", default=list(MODELS.keys()),
                        help="Models to benchmark")
    parser.add_argument("--output", default="arm_test/benchmark_results.json")
    args = parser.parse_args()

    image_tensor = load_image(args.image)
    print(f"Image: {args.image} -> {list(image_tensor.shape)}")

    labels = load_labels()

    results = []
    for model_name in args.models:
        if model_name not in MODELS:
            print(f"Unknown model: {model_name}, skipping")
            continue
        print(f"\n{'='*60}")
        print(f"Benchmarking: {model_name}")
        print(f"{'='*60}")
        try:
            r = benchmark_model(model_name, image_tensor, labels)
            results.append(r)
            print(f"  Float:  {r['float_label']} (idx={r['float_argmax']})")
            print(f"  Quant:  {r['quant_label']} (idx={r['quant_argmax']})")
            print(f"  Match:  {'YES' if r['argmax_match'] else 'NO'}")
            print(f"  Top-5 overlap: {r['top5_overlap']}/5")
            print(f"  Cosine sim:    {r['cosine_similarity']:.6f}")
            print(f"  Max abs error: {r['max_abs_error']:.4f}")
            print(f"  Mean abs error:{r['mean_abs_error']:.4f}")
            print(f"  KL divergence: {r['kl_divergence']:.6f}")
            print(f"  Float top-5:   {r['float_top5_labels']}")
            print(f"  Quant top-5:   {r['quant_top5_labels']}")
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()

    if results:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {out_path}")

        print(f"\n{'='*60}")
        print("Summary")
        print(f"{'='*60}")
        matches = sum(1 for r in results if r["argmax_match"])
        print(f"Argmax match:     {matches}/{len(results)}")
        avg_cos = np.mean([r["cosine_similarity"] for r in results])
        print(f"Avg cosine sim:   {avg_cos:.6f}")
        avg_top5 = np.mean([r["top5_overlap"] for r in results])
        print(f"Avg top-5 overlap:{avg_top5:.1f}/5")


if __name__ == "__main__":
    main()
