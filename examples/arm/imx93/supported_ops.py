#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
#
# Ethos-U65 supported operators via ExecuTorch ARM backend.
# Generated from backends/arm/operator_support/ source code.
# Use: python examples/arm/imx93/supported_ops.py

ETHOS_U65_INT_OPS = {
    "arithmetic": [
        "abs", "add", "sub", "mul", "neg", "maximum", "minimum",
        "reciprocal", "pow.Tensor_Scalar", "pow.Tensor_Tensor",
    ],
    "activation": [
        "relu", "sigmoid", "tanh", "hardtanh", "hardsigmoid",
        "hardswish", "gelu", "silu", "elu",
    ],
    "convolution": [
        "conv2d",  # regular + depthwise + grouped
        "conv_transpose2d",  # with constraints
    ],
    "pooling": [
        "avg_pool2d", "max_pool2d", "max_pool2d_with_indices",
        "_adaptive_avg_pool2d",
    ],
    "linear": ["mm", "bmm", "linear"],
    "reduction": ["amax", "amin", "any", "mean.dim", "cumsum"],
    "comparison": ["eq", "ge", "gt", "le", "lt"],
    "logical": [
        "logical_and", "logical_or", "logical_xor", "logical_not",
    ],
    "shape": [
        "cat", "expand_copy", "repeat", "view_copy", "unsqueeze_copy",
        "squeeze_copy.dims", "permute_copy", "select_copy.int",
        "split_copy.Tensor", "split_with_sizes_copy",
        "alias_copy", "copy", "detach_copy",
    ],
    "creation": [
        "full", "full_like", "arange", "eye", "linspace",
        "constant_pad_nd",
    ],
    "quantization": [
        "quantize_per_tensor", "quantize_per_channel",
        "dequantize_per_tensor", "dequantize_per_channel",
    ],
    "math": [
        "exp", "expm1", "log", "log1p", "ceil", "floor",
        "erf", "sign", "rsqrt", "clamp",
    ],
    "trig": [
        "sin", "cos", "tan", "asin", "acos", "atan",
        "sinh", "cosh", "atanh", "asinh", "acosh",
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

# Common model architectures and their Ethos-U65 delegation %
MODEL_DELEGATION = {
    "MobileNetV2": "100% NPU (3.5M params, ~3.3MB PTE)",
    "MobileNetV3": "100% NPU (2.5M params, ~2.4MB PTE)",
    "ResNet-18": "100% NPU (11.7M params, ~10MB PTE)",
    "ResNet-50": "100% NPU (25.6M params, ~22MB PTE)",
    "InceptionV3": "100% NPU (27.2M params, ~21MB PTE)",
    "ViT": "Fails Vela compilation (attention/LayerNorm issues)",
    "DeiT-Tiny": "Requires timm package",
}


def print_ops():
    print("Ethos-U65 Supported Operators (INT profile)")
    print("=" * 50)
    total = 0
    for category, ops in ETHOS_U65_INT_OPS.items():
        print(f"\n{category} ({len(ops)}):")
        for op in ops:
            print(f"  - {op}")
        total += len(ops)
    print(f"\nTotal INT ops: {total}")

    print("\n\nAdditional FP-only operators")
    print("=" * 50)
    extra = 0
    for category, ops in ETHOS_U65_FP_EXTRA_OPS.items():
        print(f"\n{category} ({len(ops)}):")
        for op in ops:
            print(f"  - {op}")
        extra += len(ops)
    print(f"\nTotal FP-only extra ops: {extra}")

    print("\n\nModel delegation results")
    print("=" * 50)
    for model, status in MODEL_DELEGATION.items():
        print(f"  {model}: {status}")


if __name__ == "__main__":
    print_ops()
