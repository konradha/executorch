#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations


ETHOS_U65_INT_OPS = {
    "arithmetic": [
        "abs",
        "add",
        "sub",
        "mul",
        "neg",
        "maximum",
        "minimum",
        "reciprocal",
        "pow.Tensor_Scalar",
        "pow.Tensor_Tensor",
    ],
    "activation": [
        "relu",
        "sigmoid",
        "tanh",
        "hardtanh",
        "hardsigmoid",
        "hardswish",
        "gelu",
        "silu",
        "elu",
    ],
    "convolution": [
        "conv2d",
        "conv_transpose2d",
    ],
    "pooling": [
        "avg_pool2d",
        "max_pool2d",
        "max_pool2d_with_indices",
        "_adaptive_avg_pool2d",
    ],
    "linear": ["mm", "bmm", "linear"],
    "reduction": ["amax", "amin", "any", "mean.dim", "cumsum"],
    "comparison": ["eq", "ge", "gt", "le", "lt"],
    "logical": ["logical_and", "logical_or", "logical_xor", "logical_not"],
    "shape": [
        "cat",
        "expand_copy",
        "repeat",
        "view_copy",
        "unsqueeze_copy",
        "squeeze_copy.dims",
        "permute_copy",
        "select_copy.int",
        "split_copy.Tensor",
        "split_with_sizes_copy",
        "alias_copy",
        "copy",
        "detach_copy",
    ],
    "creation": [
        "full",
        "full_like",
        "arange",
        "eye",
        "linspace",
        "constant_pad_nd",
    ],
    "quantization": [
        "quantize_per_tensor",
        "quantize_per_channel",
        "dequantize_per_tensor",
        "dequantize_per_channel",
    ],
    "math": [
        "exp",
        "expm1",
        "log",
        "log1p",
        "ceil",
        "floor",
        "erf",
        "sign",
        "rsqrt",
        "clamp",
    ],
    "trig": [
        "sin",
        "cos",
        "tan",
        "asin",
        "acos",
        "atan",
        "sinh",
        "cosh",
        "atanh",
        "asinh",
        "acosh",
    ],
    "other": ["index_put", "masked_fill.Scalar", "remainder.Tensor"],
}

ETHOS_U65_FP_EXTRA_OPS = {
    "normalization": [
        "native_batch_norm (no training)",
        "native_layer_norm",
        "native_group_norm",
    ],
    "activation": ["leaky_relu", "logit", "glu"],
    "softmax": ["_softmax", "_log_softmax"],
    "arithmetic": ["div", "addmm", "floor_divide"],
    "upsampling": [
        "upsample_nearest2d.vec",
        "upsample_bilinear2d.vec",
    ],
    "statistical": ["mean", "var"],
}

MODEL_DELEGATION = {
    "MobileNetV2": "100% NPU (3.5M params, ~3.3MB PTE)",
    "MobileNetV3": "100% NPU (2.5M params, ~2.4MB PTE)",
    "ResNet-18": "100% NPU (11.7M params, ~10MB PTE)",
    "ResNet-50": "100% NPU (25.6M params, ~22MB PTE)",
    "InceptionV3": "100% NPU (27.2M params, ~21MB PTE)",
    "ViT": "Fails Vela compilation (attention/LayerNorm issues)",
    "DeiT-Tiny": "Requires timm package",
}

UNDER_DOCUMENTED_TFLITE_OPS = (
    "ARG_MAX",
    "CONCATENATION",
    "DEPTHWISE_CONV_2D",
    "LOGISTIC",
    "PAD",
    "PRELU",
    "RESHAPE",
    "RESIZE_BILINEAR",
    "RESIZE_NEAREST_NEIGHBOR",
    "SLICE",
    "SPLIT",
    "SQUEEZE",
    "STRIDED_SLICE",
    "TRANSPOSE",
    "UNPACK",
)

ARM_TFLITE_CONSTRAINT_HINTS = {
    "argmax": "Depth-axis only; IFM depth must stay <= 127; output must be int32/int64.",
    "cat": "Concat axis must be valid and all non-concat dimensions must match.",
    "depthwise_conv2d": "Weights must be constant 8-bit; dilated kernel area must stay <= 4096.",
    "logistic": "Generic Ethos-U55/U65 tensor and quantization constraints apply.",
    "pad": "Padding tensor must be constant with shape [3,2] or [4,2]; only H/W/C padding is allowed.",
    "prelu": "Generic Ethos-U55/U65 tensor and quantization constraints apply.",
    "reshape": "Input/output quantization and element counts must match; target shape must be constant.",
    "resize_bilinear": "Spatial scale must be 1x/2x/4x/8x with matching width/height rules; size tensor must match OFM.",
    "resize_nearest_neighbor": "Spatial scale must be 1x/2x/4x/8x with matching width/height rules; size tensor must match OFM.",
    "slice": "Begin and size tensors must be constant.",
    "split": "Axis must be valid and divisible by the number of splits.",
    "squeeze": "Input/output quantization and element counts must match.",
    "strided_slice": "Begin/end/stride tensors must be constant; batch/channel strides must be 1.",
    "transpose": "Only a small set of rank-2/3/4 permutations are supported on U55/U65.",
    "unpack": "Generic Ethos-U55/U65 tensor and quantization constraints apply.",
}


def summarize_ops() -> dict[str, int]:
    int_ops = sum(len(ops) for ops in ETHOS_U65_INT_OPS.values())
    fp_extra_ops = sum(len(ops) for ops in ETHOS_U65_FP_EXTRA_OPS.values())
    return {
        "int_categories": len(ETHOS_U65_INT_OPS),
        "int_ops": int_ops,
        "fp_extra_categories": len(ETHOS_U65_FP_EXTRA_OPS),
        "fp_extra_ops": fp_extra_ops,
        "models": len(MODEL_DELEGATION),
        "under_documented_tflite_ops": len(UNDER_DOCUMENTED_TFLITE_OPS),
        "benchmark_hints": len(ARM_TFLITE_CONSTRAINT_HINTS),
    }


def print_ops() -> None:
    summary = summarize_ops()
    print("Ethos-U65 supported operators for the i.MX93 fast path")
    print("=" * 56)
    print(f"INT ops: {summary['int_ops']} across {summary['int_categories']} categories")
    print(
        f"FP-only extras: {summary['fp_extra_ops']} across "
        f"{summary['fp_extra_categories']} categories"
    )
    print(f"Model notes: {summary['models']}")
    print(
        "Under-documented TFLite ops: "
        f"{summary['under_documented_tflite_ops']}"
    )
    for category, ops in ETHOS_U65_INT_OPS.items():
        print(f"\n[{category}]")
        for op in ops:
            print(f"  - {op}")
    print("\n[fp-only]")
    for category, ops in ETHOS_U65_FP_EXTRA_OPS.items():
        print(f"  {category}: {', '.join(ops)}")
    print("\n[models]")
    for model, status in MODEL_DELEGATION.items():
        print(f"  {model}: {status}")
    print("\n[under-documented-tflite]")
    for op in UNDER_DOCUMENTED_TFLITE_OPS:
        print(f"  {op}")


if __name__ == "__main__":
    print_ops()
