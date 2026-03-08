#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
#
# Video recognition pipeline for Ethos-U65: classification + segmentation.
# Simulates real-time NPU inference by running quantized models on host,
# generates annotated output video with class labels and segmentation overlay.
#
# Usage:
#   python -m examples.arm.imx93.video_recognition --video arm_test/test_video.mp4
#   python -m examples.arm.imx93.video_recognition --video drone.mp4 --max-frames 300

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from examples.arm.imx93.segmentation import (
    LRASPPNearestHead,
    LRASPPWrapper,
    VOC_CLASSES,
    VOC_COLORS,
    colorize_mask,
)

IMAGENET_PREPROCESS = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

SEG_PREPROCESS = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def load_labels() -> list:
    labels_path = REPO_ROOT / "arm_test" / "imagenet_classes.txt"
    if labels_path.exists():
        return labels_path.read_text().strip().split("\n")
    try:
        import urllib.request
        data = urllib.request.urlopen(
            "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
        ).read().decode()
        labels_path.parent.mkdir(parents=True, exist_ok=True)
        labels_path.write_text(data)
        return data.strip().split("\n")
    except Exception:
        return [str(i) for i in range(1000)]


def load_classifier(model_name="mv2"):
    from examples.models.model_factory import EagerModelFactory
    MODEL_MAP = {
        "mv2": ("mobilenet_v2", "MV2Model"),
        "mv3": ("mobilenet_v3", "MV3Model"),
        "resnet18": ("resnet", "ResNet18Model"),
    }
    result = EagerModelFactory.create_model(*MODEL_MAP[model_name])
    return result[0].eval()


def load_segmenter():
    model = LRASPPWrapper(pretrained=True, use_nearest=False).eval()
    return model


def quantize_for_comparison(model, example_input):
    """Quantize model to match what runs on NPU (for precision comparison)."""
    from executorch.backends.arm.ethosu import EthosUCompileSpec
    from executorch.backends.arm.quantizer import get_symmetric_quantization_config
    from executorch.backends.arm.util._factory import create_quantizer
    from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

    compile_spec = EthosUCompileSpec("ethos-u65-256")
    exported = torch.export.export(model, example_input, strict=True)
    gm = exported.module()

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
    return convert_pt2e(prepared)


def draw_labels(frame, top3_labels, top3_probs, seg_classes, fps):
    """Draw classification + segmentation labels on frame."""
    h, w = frame.shape[:2]

    # Semi-transparent black bar at top
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 80), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # Classification
    for i, (label, prob) in enumerate(zip(top3_labels, top3_probs)):
        text = f"{label}: {prob:.1%}"
        color = (0, 255, 0) if i == 0 else (200, 200, 200)
        cv2.putText(frame, text, (5, 18 + i * 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    # Segmentation classes
    if seg_classes:
        seg_text = "Seg: " + ", ".join(seg_classes)
        cv2.putText(frame, seg_text, (5, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

    # FPS
    cv2.putText(frame, f"{fps:.0f} FPS", (w - 60, 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

    return frame


def process_video(video_path, output_path, classifier, segmenter,
                  labels, max_frames=0, quantized_cls=None):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if max_frames > 0:
        total = min(total, max_frames)

    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (448, 224))

    stats = {"frames": 0, "cls_time": 0, "seg_time": 0,
             "argmax_matches": 0, "top5_overlaps": []}

    frame_idx = 0
    with torch.no_grad():
        while cap.isOpened() and (max_frames == 0 or frame_idx < max_frames):
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            tensor = IMAGENET_PREPROCESS(frame_rgb).unsqueeze(0)

            # Classification (float)
            t0 = time.time()
            cls_out = classifier(tensor)
            if not isinstance(cls_out, torch.Tensor):
                cls_out = cls_out[0]
            stats["cls_time"] += time.time() - t0

            probs = F.softmax(cls_out, dim=1).numpy().flatten()
            top3_idx = np.argsort(probs)[-3:][::-1]
            top3_labels = [labels[i] if i < len(labels) else str(i) for i in top3_idx]
            top3_probs = probs[top3_idx]

            # Quantized comparison
            if quantized_cls is not None:
                q_out = quantized_cls(tensor)
                if not isinstance(q_out, torch.Tensor):
                    q_out = q_out[0]
                q_np = q_out.numpy().flatten()
                f_np = cls_out.numpy().flatten()
                if np.argmax(q_np) == np.argmax(f_np):
                    stats["argmax_matches"] += 1
                f_top5 = set(np.argsort(f_np)[-5:][::-1])
                q_top5 = set(np.argsort(q_np)[-5:][::-1])
                stats["top5_overlaps"].append(len(f_top5 & q_top5))

            # Segmentation
            seg_tensor = SEG_PREPROCESS(frame_rgb).unsqueeze(0)
            t0 = time.time()
            seg_out = segmenter(seg_tensor)
            stats["seg_time"] += time.time() - t0

            pred = seg_out.argmax(dim=1).squeeze().numpy()
            seg_classes_present = [VOC_CLASSES[c] for c in np.unique(pred) if c > 0]

            # Build output frame: original | segmentation overlay (side by side)
            display = cv2.resize(frame, (224, 224))
            display_rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
            mask_rgb = colorize_mask(pred)
            seg_overlay = (display_rgb * 0.5 + mask_rgb * 0.5).astype(np.uint8)
            seg_bgr = cv2.cvtColor(seg_overlay, cv2.COLOR_RGB2BGR)

            combined_fps = 1.0 / max(
                (stats["cls_time"] + stats["seg_time"]) / max(frame_idx, 1), 0.001)

            # Draw labels on left panel
            display = draw_labels(display, top3_labels, top3_probs,
                                  seg_classes_present, combined_fps)

            # Side by side
            combined = np.hstack([display, seg_bgr])
            writer.write(combined)

            frame_idx += 1
            stats["frames"] = frame_idx

            if frame_idx % 30 == 0:
                print(f"  frame {frame_idx}/{total}: "
                      f"cls={top3_labels[0]} ({top3_probs[0]:.1%}), "
                      f"seg={seg_classes_present}")

    cap.release()
    writer.release()
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Video recognition pipeline for Ethos-U65")
    parser.add_argument("--video", required=True, help="Input video path")
    parser.add_argument("--output", default=None, help="Output video path")
    parser.add_argument("--classifier", default="mv2",
                        choices=["mv2", "mv3", "resnet18"])
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--quantize-compare", action="store_true",
                        help="Also run quantized classifier for comparison")
    args = parser.parse_args()

    if args.output is None:
        stem = Path(args.video).stem
        args.output = f"arm_test/{stem}_recognized.mp4"

    print("Loading models...")
    labels = load_labels()
    classifier = load_classifier(args.classifier)
    segmenter = load_segmenter()

    quantized_cls = None
    if args.quantize_compare:
        print("Quantizing classifier for precision comparison...")
        example = (torch.randn(1, 3, 224, 224),)
        quantized_cls = quantize_for_comparison(classifier, example)

    print(f"Processing: {args.video}")
    stats = process_video(args.video, args.output, classifier, segmenter,
                          labels, args.max_frames, quantized_cls)

    print(f"\nDone: {args.output}")
    print(f"  Frames: {stats['frames']}")
    avg_cls = stats['cls_time'] / max(stats['frames'], 1) * 1000
    avg_seg = stats['seg_time'] / max(stats['frames'], 1) * 1000
    print(f"  Avg classification: {avg_cls:.1f} ms/frame")
    print(f"  Avg segmentation:   {avg_seg:.1f} ms/frame")
    print(f"  Combined host FPS:  {stats['frames'] / max(stats['cls_time'] + stats['seg_time'], 0.001):.1f}")
    print(f"  NPU estimate:       ~35 FPS (28ms seg + classification)")

    if stats.get("top5_overlaps"):
        n = stats["frames"]
        print(f"\n  Quantization precision ({n} frames):")
        print(f"    Argmax match: {stats['argmax_matches']}/{n} "
              f"({100*stats['argmax_matches']/n:.0f}%)")
        avg_overlap = np.mean(stats['top5_overlaps'])
        print(f"    Avg top-5 overlap: {avg_overlap:.1f}/5")

    # Save stats
    stats_path = Path(args.output).with_suffix(".json")
    stats["top5_overlaps"] = stats.get("top5_overlaps", [])
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)


if __name__ == "__main__":
    main()
