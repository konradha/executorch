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

ETHOS_U55_U65_GENERIC_CONSTRAINTS = (
    "All required operator attributes must be specified.",
    "Input and output tensors must be static and have defined shapes.",
    "Output tensors cannot be scalar except for QUANTIZE.",
    "Input and output tensors must be at most 4D.",
    "Tensors must use int16, int32, int8, or uint8 except for ARG_MAX outputs.",
    "Input, output, and weight tensors must have quantization parameters except for ARG_MAX, MIRROR_PAD, SHAPE, and TRANSPOSE.",
    "Tensor dimensions must stay in [1, 65535].",
    "Per-axis quantization is only supported for CONV_2D, DEPTHWISE_CONV_2D, FULLY_CONNECTED, and TRANSPOSE_CONV.",
    "IFM batch size must be 1 except for FULLY_CONNECTED, RESHAPE, SHAPE, SLICE, SOFTMAX, SPLIT, SPLIT_V, SQUEEZE, STRIDED_SLICE, and UNPACK.",
    "Fused activations are limited to LOGISTIC, RELU, RELU6, RELU_0_TO_1, RELU_N1_TO_1, and TANH.",
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

IMX93_FASTPATH_MATRIX = {
    "argmax": {
        "tflite_op": "ARG_MAX",
        "official_constraints": (
            "IFM must be int8 or uint8.",
            "OFM must be int32 or int64.",
            "Reduction must run along the depth axis.",
            "IFM depth must be <= 127.",
        ),
        "observed_status": "not_delegated",
        "observed_sizes": (8, 16),
        "observed_notes": "Current fastpath export does not emit a vela_model block for this case; exact device results are from the CPU path.",
    },
    "cat": {
        "tflite_op": "CONCATENATION",
        "official_constraints": (
            "Axis attribute must exist.",
            "Axis must be in [0, rank(ofm)).",
            "All input ranks must match OFM rank.",
            "All non-concat dimensions must match.",
            "OFM concat dimension must equal the sum of IFM concat dimensions.",
        ),
        "observed_status": "correct",
        "observed_sizes": (16,),
        "observed_notes": "Delegates and runs correctly when exported with channels-last 4D tensors and compared using raw tensor storage layout; the size-16 spot check landed at RMSE about 0.0025.",
    },
    "depthwise_conv2d": {
        "tflite_op": "DEPTHWISE_CONV_2D",
        "official_constraints": (
            "Stride and dilation values must be integer typed.",
            "Dilated kernel height must be in [1, 64].",
            "Dilated kernel area must be in [1, 4096].",
            "Weights must be constant 8-bit tensors.",
            "Sum of weights must not exceed 8323072.",
            "Optional bias must be 1D int32 or int64 with values fitting in 40 bits.",
            "Stride width and height must each be between 1 and 3.",
            "For depth_multiplier > 1, IFM channels must be 1 and OFM channels must equal the depth multiplier.",
        ),
        "observed_status": "correct",
        "observed_sizes": (16,),
        "observed_notes": "Delegates and runs correctly when exported with channels-last 4D tensors and compared using raw tensor storage layout; the size-16 spot check landed at RMSE about 0.0011.",
    },
    "logistic": {
        "tflite_op": "LOGISTIC",
        "official_constraints": ETHOS_U55_U65_GENERIC_CONSTRAINTS,
        "observed_status": "correct",
        "observed_sizes": (16,),
        "observed_notes": "Delegates and runs correctly when exported with channels-last 4D tensors and compared using raw tensor storage layout; the size-16 spot check landed at RMSE about 0.0019.",
    },
    "pad": {
        "tflite_op": "PAD",
        "official_constraints": (
            "Exactly 2 inputs are required.",
            "Padding tensor must be constant.",
            "Padding tensor must have shape [3,2] or [4,2].",
            "Padding tensor must be int32 or int64.",
            "Padding may affect only height, width, and depth.",
        ),
        "observed_status": "correct",
        "observed_sizes": (16,),
        "observed_notes": "Delegates and runs on device; use channels-last 4D export for correct tensor storage ordering.",
    },
    "prelu": {
        "tflite_op": "PRELU",
        "official_constraints": ETHOS_U55_U65_GENERIC_CONSTRAINTS,
        "observed_status": "not_delegated",
        "observed_sizes": (8, 16),
        "observed_notes": "Current fastpath export does not emit a vela_model block for this case; exact device results are from the CPU path.",
    },
    "reshape": {
        "tflite_op": "RESHAPE",
        "official_constraints": (
            "Input and output quantization must match.",
            "Input and output element counts must match.",
            "Target shape must be constant.",
        ),
        "observed_status": "not_delegated",
        "observed_sizes": (8, 16, 32),
        "observed_notes": "Current fastpath export does not emit a vela_model block for this case, so device results are from the CPU path.",
    },
    "resize_bilinear": {
        "tflite_op": "RESIZE_BILINEAR",
        "official_constraints": (
            "IFM/OFM width and height must either match, be 1, or scale by 1x/2x/4x/8x under the align_corners rules.",
            "Size tensor must match the OFM shape.",
            "align_corners and half_pixel_centers cannot both be True.",
            "For half_pixel_centers, IFM width and height must be 1 or OFM must be exactly 2x IFM.",
        ),
        "observed_status": "export_failed",
        "observed_sizes": (8, 16),
        "observed_notes": "Current i.MX93 export path fails inside Vela/regor before device execution.",
    },
    "resize_nearest_neighbor": {
        "tflite_op": "RESIZE_NEAREST_NEIGHBOR",
        "official_constraints": (
            "IFM/OFM width and height must either match, be 1, or scale by 1x/2x/4x/8x under the align_corners rules.",
            "Size tensor must match the OFM shape.",
            "align_corners and half_pixel_centers cannot both be True.",
        ),
        "observed_status": "export_failed",
        "observed_sizes": (8, 16),
        "observed_notes": "Current i.MX93 export path fails inside Vela/regor before device execution.",
    },
    "slice": {
        "tflite_op": "SLICE",
        "official_constraints": (
            "Begin and size tensors must be constant.",
        ),
        "observed_status": "correct",
        "observed_sizes": (16,),
        "observed_notes": "Delegates and runs on device; use channels-last 4D export for correct tensor storage ordering.",
    },
    "split": {
        "tflite_op": "SPLIT",
        "official_constraints": (
            "Axis must be in [-rank(ifm), rank(ifm)).",
            "Axis must be divisible by the number of splits.",
        ),
        "observed_status": "correct",
        "observed_sizes": (16,),
        "observed_notes": "Delegates and runs correctly when exported with channels-last 4D tensors and compared using raw tensor storage layout; the size-16 spot check landed at RMSE about 0.0012 across both outputs.",
    },
    "squeeze": {
        "tflite_op": "SQUEEZE",
        "official_constraints": (
            "Input and output quantization must match.",
            "Input and output element counts must match.",
        ),
        "observed_status": "correct",
        "observed_sizes": (16,),
        "observed_notes": "Delegates and runs on device; use channels-last 4D export for correct tensor storage ordering.",
    },
    "strided_slice": {
        "tflite_op": "STRIDED_SLICE",
        "official_constraints": (
            "Exactly 4 input tensors are required.",
            "Begin, end, and stride tensors must be constant.",
            "ellipsis_mask must be 0.",
            "new_axis_mask and shrink_axis_mask cannot both be set.",
            "Slice end values must exceed begin values.",
            "Batch and channel strides must be 1.",
            "Offset attribute must be False.",
        ),
        "observed_status": "runtime_failed",
        "observed_sizes": (16,),
        "observed_notes": "Delegates, but current device execution hangs and leaves the remoteproc path needing a restart.",
    },
    "transpose": {
        "tflite_op": "TRANSPOSE",
        "official_constraints": (
            "Permutation tensor must be constant 1D with rank(ifm) elements.",
            "Permutation values must be in [0, rank(ifm)).",
            "Only a small set of rank-2/3/4 permutations are supported on U55/U65.",
        ),
        "observed_status": "correct",
        "observed_sizes": (16,),
        "observed_notes": "Delegates and runs on device for the exercised permutation; channels-last 4D export keeps tensor storage aligned with the driver.",
    },
    "unpack": {
        "tflite_op": "UNPACK",
        "official_constraints": ETHOS_U55_U65_GENERIC_CONSTRAINTS,
        "observed_status": "runtime_failed",
        "observed_sizes": (8, 16),
        "observed_notes": "Delegates, but device execution currently fails with IOCTL failure.",
    },
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
        "fastpath_matrix": len(IMX93_FASTPATH_MATRIX),
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
    print("\n[imx93-fastpath-matrix]")
    for op_name, row in IMX93_FASTPATH_MATRIX.items():
        print(
            f"  {op_name}: {row['observed_status']} "
            f"sizes={row['observed_sizes']} "
            f"op={row['tflite_op']}"
        )


if __name__ == "__main__":
    print_ops()
