#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
#
# Video preprocessing and reconstruction for on-device NPU inference.
#
# Phase 1 (preprocess): Convert video frames to raw float32 tensors.
# Phase 2 (reconstruct): Read NPU output tensors, annotate frames, write video.
#
# Usage:
#   # Preprocess: video -> raw frame tensors
#   python -m examples.arm.imx93.preprocess_video preprocess \
#       --video drone.mp4 --output-dir arm_test/video_frames/ \
#       --model mv2 --max-frames 300
#
#   # ... run video_driver on device ...
#
#   # Reconstruct: NPU outputs -> annotated video
#   python -m examples.arm.imx93.preprocess_video reconstruct \
#       --video drone.mp4 --output-dir arm_test/video_frames/ \
#       --results-dir arm_test/video_results/ \
#       --model mv2 --output-video arm_test/drone_npu.mp4

import argparse
import csv
import json
import struct
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision import transforms

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

# ImageNet normalization
IMAGENET_PREPROCESS = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# Models and their output interpretation
MODEL_INFO = {
    "mv2": {"task": "classification", "input": [1, 3, 224, 224], "classes": 1000},
    "mv3": {"task": "classification", "input": [1, 3, 224, 224], "classes": 1000},
    "resnet18": {"task": "classification", "input": [1, 3, 224, 224], "classes": 1000},
    "resnet50": {"task": "classification", "input": [1, 3, 224, 224], "classes": 1000},
    "ic3": {"task": "classification", "input": [1, 3, 224, 224], "classes": 1000},
    "lraspp": {"task": "segmentation", "input": [1, 3, 224, 224], "classes": 21},
    "large_convnet": {"task": "classification", "input": [1, 3, 64, 64], "classes": 1000},
}

VOC_CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
    "car", "cat", "chair", "cow", "dining table", "dog", "horse",
    "motorbike", "person", "potted plant", "sheep", "sofa", "train", "tv"
]

VOC_COLORS = np.array([
    [0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
    [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0], [192, 0, 0],
    [64, 128, 0], [192, 128, 0], [64, 0, 128], [192, 0, 128], [64, 128, 128],
    [192, 128, 128], [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
    [0, 64, 128],
], dtype=np.uint8)


def load_imagenet_labels():
    labels_path = REPO_ROOT / "arm_test" / "imagenet_classes.txt"
    if labels_path.exists():
        return labels_path.read_text().strip().split("\n")
    return [str(i) for i in range(1000)]


def preprocess(args):
    """Convert video frames to raw float32 tensors for device inference."""
    info = MODEL_INFO[args.model]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    max_frames = args.max_frames if args.max_frames > 0 else total

    input_shape = info["input"]
    input_bytes = int(np.prod(input_shape)) * 4  # float32

    # Also save original frames for reconstruction
    orig_dir = out_dir / "originals"
    orig_dir.mkdir(exist_ok=True)

    frame_idx = 0
    while cap.isOpened() and frame_idx < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Save original (resized to 224x224 for overlay)
        orig_resized = cv2.resize(frame, (224, 224))
        cv2.imwrite(str(orig_dir / f"frame_{frame_idx:04d}.png"), orig_resized)

        # Preprocess to tensor
        if args.model == "large_convnet":
            # 64x64 input
            pil_transforms = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(64),
                transforms.CenterCrop(64),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])
            tensor = pil_transforms(frame_rgb).unsqueeze(0)
        else:
            tensor = IMAGENET_PREPROCESS(frame_rgb).unsqueeze(0)

        # Save raw float32 binary
        raw = tensor.numpy().tobytes()
        assert len(raw) == input_bytes, f"Expected {input_bytes}, got {len(raw)}"
        frame_path = out_dir / f"frame_{frame_idx:04d}.bin"
        with open(frame_path, "wb") as f:
            f.write(raw)

        frame_idx += 1
        if frame_idx % 50 == 0:
            print(f"  preprocessed {frame_idx}/{max_frames}")

    cap.release()

    # Save metadata
    meta = {
        "model": args.model,
        "video": args.video,
        "fps": fps,
        "num_frames": frame_idx,
        "input_shape": input_shape,
        "input_bytes": input_bytes,
        "output_bytes": int(np.prod(info["input"][:1] +
                           ([info["classes"]] if info["task"] == "classification"
                            else [info["classes"]] + input_shape[2:]))) * 4,
        "task": info["task"],
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Preprocessed {frame_idx} frames -> {out_dir}")
    print(f"  Input tensor: {input_shape} = {input_bytes} bytes/frame")
    print(f"  Total data: {frame_idx * input_bytes / 1024 / 1024:.1f} MB")
    print(f"\nNext: copy {out_dir} to device and run video_driver")


def reconstruct(args):
    """Read NPU outputs and create annotated video."""
    out_dir = Path(args.output_dir)
    results_dir = Path(args.results_dir)
    info = MODEL_INFO[args.model]

    # Load metadata
    meta_path = out_dir / "meta.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        fps = meta["fps"]
        num_frames = meta["num_frames"]
    else:
        fps = 30.0
        num_frames = len(list(results_dir.glob("out_*.bin")))

    # Load timing data
    timing = {}
    timing_path = results_dir / "timing.csv"
    if timing_path.exists():
        with open(timing_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                timing[int(row["frame"])] = {
                    "wall_ms": float(row["wall_ms"]),
                    "m33_cycles": int(row["m33_cycles"]),
                    "argmax": int(row["argmax"]),
                }

    labels = load_imagenet_labels()
    orig_dir = out_dir / "originals"

    if info["task"] == "segmentation":
        out_size = (448, 224)  # side by side
    else:
        out_size = (224, 224)

    writer = cv2.VideoWriter(args.output_video,
                             cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, out_size)

    total_wall = 0
    for f in range(num_frames):
        out_path = results_dir / f"out_{f:04d}.bin"
        orig_path = orig_dir / f"frame_{f:04d}.png"

        if not out_path.exists():
            continue

        # Read original frame
        if orig_path.exists():
            frame = cv2.imread(str(orig_path))
        else:
            frame = np.zeros((224, 224, 3), dtype=np.uint8)

        # Read output tensor
        with open(out_path, "rb") as fh:
            raw = fh.read()
        out_np = np.frombuffer(raw, dtype=np.float32)

        t = timing.get(f, {})
        wall_ms = t.get("wall_ms", 0)
        npu_fps = 1000.0 / wall_ms if wall_ms > 0 else 0
        total_wall += wall_ms

        if info["task"] == "classification":
            # Top-3 predictions
            top3_idx = np.argsort(out_np)[-3:][::-1]
            # Softmax for display
            exp_out = np.exp(out_np - np.max(out_np))
            probs = exp_out / exp_out.sum()

            # Draw on frame
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (224, 70), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

            for i, idx in enumerate(top3_idx):
                label = labels[idx] if idx < len(labels) else str(idx)
                text = f"{label}: {probs[idx]:.1%}"
                color = (0, 255, 0) if i == 0 else (200, 200, 200)
                cv2.putText(frame, text, (5, 18 + i * 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

            cv2.putText(frame, f"{npu_fps:.0f} FPS", (170, 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            writer.write(frame)

        elif info["task"] == "segmentation":
            # Reshape to [21, 224, 224] and get class predictions
            n_classes = info["classes"]
            h, w = 224, 224
            out_map = out_np.reshape(n_classes, h, w)
            pred = np.argmax(out_map, axis=0)

            # Colorize
            mask = np.zeros((h, w, 3), dtype=np.uint8)
            for c in range(len(VOC_COLORS)):
                mask[pred == c] = VOC_COLORS[c]

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            blended = (frame_rgb * 0.5 + mask * 0.5).astype(np.uint8)
            seg_bgr = cv2.cvtColor(blended, cv2.COLOR_RGB2BGR)

            # Labels
            classes_found = [VOC_CLASSES[c] for c in np.unique(pred) if c > 0]
            cv2.putText(frame, f"{npu_fps:.0f} FPS", (170, 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            if classes_found:
                cv2.putText(frame, ", ".join(classes_found[:3]), (5, 215),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1)

            combined = np.hstack([frame, seg_bgr])
            writer.write(combined)

        if f % 30 == 0 and f > 0:
            avg_fps = f / (total_wall / 1000.0) if total_wall > 0 else 0
            print(f"  reconstructed {f}/{num_frames} (avg {avg_fps:.1f} NPU FPS)")

    writer.release()

    avg_ms = total_wall / max(num_frames, 1)
    avg_fps = 1000.0 / avg_ms if avg_ms > 0 else 0
    print(f"\nReconstructed {num_frames} frames -> {args.output_video}")
    print(f"  Avg inference: {avg_ms:.1f} ms/frame")
    print(f"  NPU FPS:       {avg_fps:.1f}")
    print(f"  Video FPS:     {fps}")
    if avg_fps >= fps:
        print(f"  REAL-TIME: NPU ({avg_fps:.0f}) >= video ({fps:.0f})")
    else:
        print(f"  NOT REAL-TIME: NPU ({avg_fps:.0f}) < video ({fps:.0f})")


def main():
    parser = argparse.ArgumentParser(
        description="Video preprocessing and reconstruction for NPU inference")
    sub = parser.add_subparsers(dest="command")

    p_pre = sub.add_parser("preprocess", help="Convert video to raw tensors")
    p_pre.add_argument("--video", required=True)
    p_pre.add_argument("--output-dir", required=True)
    p_pre.add_argument("--model", default="mv2", choices=MODEL_INFO.keys())
    p_pre.add_argument("--max-frames", type=int, default=0)

    p_rec = sub.add_parser("reconstruct", help="NPU outputs -> annotated video")
    p_rec.add_argument("--video", required=True, help="Original video (for frames)")
    p_rec.add_argument("--output-dir", required=True, help="Preprocessed frames dir")
    p_rec.add_argument("--results-dir", required=True, help="NPU output dir")
    p_rec.add_argument("--model", default="mv2", choices=MODEL_INFO.keys())
    p_rec.add_argument("--output-video", required=True)

    args = parser.parse_args()
    if args.command == "preprocess":
        preprocess(args)
    elif args.command == "reconstruct":
        reconstruct(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
