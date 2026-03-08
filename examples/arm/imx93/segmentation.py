#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
#
# Image segmentation on Ethos-U65 using LRASPP-MobileNetV3.
# Exports model, runs host-side inference on images/video frames,
# and visualizes segmentation masks.
#
# Usage:
#   python -m examples.arm.imx93.segmentation --export
#   python -m examples.arm.imx93.segmentation --image dog.jpg
#   python -m examples.arm.imx93.segmentation --video drone_footage.mp4 --output seg_output.mp4

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models.segmentation import lraspp_mobilenet_v3_large

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

# Pascal VOC class names and colors
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


class LRASPPWrapper(nn.Module):
    """Wrapper that returns just the segmentation tensor (not OrderedDict).

    For NPU export, use_nearest=True replaces bilinear upsampling with nearest
    (bilinear crashes Vela compiler; nearest is in INT profile).
    """

    def __init__(self, pretrained=True, use_nearest=False):
        super().__init__()
        weights = "DEFAULT" if pretrained else None
        self.model = lraspp_mobilenet_v3_large(weights=weights, num_classes=21)
        self.use_nearest = use_nearest

    def forward(self, x):
        import torch.nn.functional as F
        features = self.model.backbone(x)
        out = self.model.classifier(features)
        if self.use_nearest:
            out = F.interpolate(out, size=x.shape[-2:], mode="nearest")
        else:
            out = F.interpolate(out, size=x.shape[-2:], mode="bilinear",
                                align_corners=False)
        return out


class LRASPPNearestHead(nn.Module):
    """LRASPP head with nearest upsampling instead of bilinear."""

    def __init__(self, original_head):
        super().__init__()
        self.cbr = original_head.cbr
        self.scale = original_head.scale
        self.low_classifier = original_head.low_classifier
        self.high_classifier = original_head.high_classifier

    def forward(self, input):
        import torch.nn.functional as F
        low = input["low"]
        high = input["high"]
        x = self.cbr(high)
        s = self.scale(high)
        x = x * s
        x = F.interpolate(x, size=low.shape[-2:], mode="nearest")
        return self.low_classifier(low) + self.high_classifier(x)


PREPROCESS = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def load_image(path: str) -> torch.Tensor:
    from PIL import Image
    img = Image.open(path).convert("RGB")
    return PREPROCESS(img).unsqueeze(0)


def segment_image(model, image_tensor):
    """Run segmentation and return class map + detected classes."""
    with torch.no_grad():
        out = model(image_tensor)
    pred = out.argmax(dim=1).squeeze().numpy()
    classes_present = np.unique(pred)
    detected = [(int(c), VOC_CLASSES[c]) for c in classes_present if c > 0]
    return pred, detected


def colorize_mask(pred):
    """Convert class predictions to RGB color mask."""
    h, w = pred.shape
    color = np.zeros((h, w, 3), dtype=np.uint8)
    for c in range(len(VOC_COLORS)):
        color[pred == c] = VOC_COLORS[c]
    return color


def overlay(image_path, pred, alpha=0.5):
    """Overlay segmentation mask on original image."""
    from PIL import Image
    img = Image.open(image_path).convert("RGB")
    img = img.resize((224, 224))
    img_np = np.array(img)
    mask_rgb = colorize_mask(pred)
    blended = (img_np * (1 - alpha) + mask_rgb * alpha).astype(np.uint8)
    return Image.fromarray(blended)


def export_model(output_dir: str, target: str = "ethos-u65-256"):
    """Export LRASPP for Ethos-U65."""
    from executorch.backends.arm.ethosu import EthosUCompileSpec
    from executorch.backends.arm.quantizer import get_symmetric_quantization_config
    from executorch.backends.arm.util._factory import create_partitioner, create_quantizer
    from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower
    from executorch.extension.export_util.utils import save_pte_program
    from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    model = LRASPPWrapper(pretrained=True, use_nearest=True).eval()
    # Replace head with nearest-upsampling version
    model.model.classifier = LRASPPNearestHead(model.model.classifier)
    example_input = (torch.randn(1, 3, 224, 224),)
    params = sum(p.numel() for p in model.parameters())
    print(f"LRASPP-MobileNetV3 (nearest): {params/1e6:.1f}M params")

    compile_spec = EthosUCompileSpec(target,
        extra_flags=["--verbose-operators", "--verbose-cycle-estimate"])

    # Export
    exported = torch.export.export(model, example_input, strict=True)
    gm = exported.module()

    # Strip _guards_fn
    for node in list(gm.graph.nodes):
        if node.op == "call_module" and node.target == "_guards_fn":
            gm.graph.erase_node(node)
            if hasattr(gm, "_guards_fn"):
                delattr(gm, "_guards_fn")
    gm.graph.lint()
    gm.recompile()

    # Quantize
    quantizer = create_quantizer(compile_spec)
    qconfig = get_symmetric_quantization_config(is_per_channel=True)
    quantizer.set_global(qconfig)
    prepared = prepare_pt2e(gm, quantizer)
    prepared(*example_input)
    quantized = convert_pt2e(prepared)

    # Re-export and lower
    exported = torch.export.export(quantized, example_input, strict=True)
    partitioner = create_partitioner(compile_spec)
    edge = to_edge_transform_and_lower(
        exported,
        partitioner=[partitioner],
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
    )

    pte_path = out / f"lraspp_mv3_{target}.pte"
    save_pte_program(edge.to_executorch(), str(pte_path))
    import os
    print(f"PTE saved: {pte_path} ({os.path.getsize(pte_path)/1024/1024:.1f} MB)")
    return str(pte_path)


def process_video(model, video_path: str, output_path: str, max_frames: int = 0):
    """Process video frames and write segmentation overlay."""
    import cv2

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if max_frames > 0:
        total = min(total, max_frames)

    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (224, 224))

    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    to_display = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
    ])

    frame_idx = 0
    all_classes = set()
    with torch.no_grad():
        while cap.isOpened() and (max_frames == 0 or frame_idx < max_frames):
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            tensor = preprocess(frame_rgb).unsqueeze(0)
            out = model(tensor)
            pred = out.argmax(dim=1).squeeze().numpy()

            classes = set(int(c) for c in np.unique(pred) if c > 0)
            all_classes |= classes

            # Overlay
            display = np.array(to_display(frame_rgb))
            mask_rgb = colorize_mask(pred)
            blended = (display * 0.5 + mask_rgb * 0.5).astype(np.uint8)
            writer.write(cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))

            frame_idx += 1
            if frame_idx % 50 == 0:
                detected = [VOC_CLASSES[c] for c in sorted(classes)]
                print(f"  frame {frame_idx}/{total}: {detected}")

    cap.release()
    writer.release()

    detected_names = [VOC_CLASSES[c] for c in sorted(all_classes)]
    print(f"Processed {frame_idx} frames -> {output_path}")
    print(f"Classes detected: {detected_names}")
    return frame_idx


def main():
    parser = argparse.ArgumentParser(description="LRASPP segmentation for Ethos-U65")
    parser.add_argument("--export", action="store_true", help="Export PTE model")
    parser.add_argument("--image", type=str, help="Segment a single image")
    parser.add_argument("--video", type=str, help="Segment video frames")
    parser.add_argument("--output", type=str, default="arm_test/models/lraspp")
    parser.add_argument("--target", default="ethos-u65-256")
    parser.add_argument("--max-frames", type=int, default=0,
                        help="Max video frames (0=all)")
    parser.add_argument("--pretrained", action="store_true", default=True)
    args = parser.parse_args()

    if args.export:
        export_model(args.output, args.target)
        return

    model = LRASPPWrapper(pretrained=args.pretrained).eval()

    if args.image:
        tensor = load_image(args.image)
        pred, detected = segment_image(model, tensor)
        print(f"Detected: {detected}")

        out_dir = Path(args.output)
        out_dir.mkdir(parents=True, exist_ok=True)
        result = overlay(args.image, pred)
        result_path = out_dir / f"{Path(args.image).stem}_segmented.png"
        result.save(str(result_path))
        print(f"Saved: {result_path}")

    if args.video:
        out_path = args.output if args.output.endswith(".mp4") else \
            str(Path(args.output) / f"{Path(args.video).stem}_segmented.mp4")
        process_video(model, args.video, out_path, args.max_frames)


if __name__ == "__main__":
    main()
