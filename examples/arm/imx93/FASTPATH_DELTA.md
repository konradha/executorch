# i.MX93 Fastpath Delta

This report compares the under-documented U65/TFLite operator set against
what the ExecuTorch i.MX93 Linux fastpath actually delivers today.
It does not describe the full model-proven CNN core by itself.

Source basis:

- Official support and constraints come from the Vela 4.5.0 `--supported-ops-report` for U55/U65.
- Fastpath status comes from this branch's export checks, delegated payload inspection, and on-device runner validation.
- The broader CNN core path is anchored by full-model delegation artifacts from MobileNet/ResNet-style exports.

Status meanings:

- `correct`: emits a delegated payload and matches device execution.
- `metadata_only`: supported as a view/bookkeeping op, not meaningful NPU work.
- `not_delegated`: officially supported in principle, but no delegated payload on this path.
- `export_failed`: blocked before device execution.
- `unsafe_for_inference`: delegates, but the i.MX93 runtime path is still wrong or unstable.

## Model-Proven Core Path

These operators are already proven by heavily or fully delegated CNN-class
models in the sibling `executorch/` checkout. This is the main performance
path for practical inference on i.MX93.

`aten_convolution_default`, `aten_add_tensor`, `aten_relu_default`, `aten_hardtanh_default`, `aten_linear_default`, `aten_mean_dim`, `aten_view_copy_default`, `dim_order_ops__clone_dim_order_default`, `quantized_decomposed_dequantize_per_channel_default`, `quantized_decomposed_dequantize_per_tensor_default`, `quantized_decomposed_quantize_per_tensor_default`

## Trusted Delegated Set

These are the under-documented ops that are both officially supported on
U65 and currently validated on the i.MX93 Linux fastpath.

`cat`, `depthwise_conv2d`, `logistic`, `pad`, `resize_nearest_neighbor`, `slice`, `split`, `squeeze`, `transpose`

## Open Delta Set

These ops are officially supported on paper but still have a real gap on
the current fastpath, or are metadata-only and should not be counted as
delegated NPU work.

`argmax`, `prelu`, `reshape`, `resize_bilinear`, `strided_slice`, `unpack`

## Clean delegated on i.MX93

| Op | TFLite op | Sizes | Payload | Gap layer | Notes |
| --- | --- | --- | --- | --- | --- |
| `cat` | `CONCATENATION` | `16` | yes | No observed gap | Delegates and runs correctly when exported with channels-last 4D tensors and compared using raw tensor storage layout; the size-16 spot check landed at RMSE about 0.0025. |
| `depthwise_conv2d` | `DEPTHWISE_CONV_2D` | `16` | yes | No observed gap | Delegates and runs correctly when exported with channels-last 4D tensors and compared using raw tensor storage layout; the size-16 spot check landed at RMSE about 0.0011. |
| `logistic` | `LOGISTIC` | `16` | yes | No observed gap | Delegates and runs correctly when exported with channels-last 4D tensors and compared using raw tensor storage layout; the size-16 spot check landed at RMSE about 0.0019. |
| `pad` | `PAD` | `16` | yes | No observed gap | Delegates and runs on device; use channels-last 4D export for correct tensor storage ordering. |
| `resize_nearest_neighbor` | `RESIZE_NEAREST_NEIGHBOR` | `8, 16` | yes | No observed gap | Delegates and runs correctly on device. The rewritten size-16 case is exact against the delegated TOSA reference (RMSE 0). |
| `slice` | `SLICE` | `16` | yes | No observed gap | Delegates and runs on device; use channels-last 4D export for correct tensor storage ordering. |
| `split` | `SPLIT` | `16` | yes | No observed gap | Delegates and runs correctly when exported with channels-last 4D tensors and compared using raw tensor storage layout; the size-16 spot check landed at RMSE about 0.0012 across both outputs. |
| `squeeze` | `SQUEEZE` | `16` | yes | No observed gap | Delegates and runs on device; use channels-last 4D export for correct tensor storage ordering. |
| `transpose` | `TRANSPOSE` | `16` | yes | No observed gap | Delegates and runs on device for the exercised permutation; channels-last 4D export keeps tensor storage aligned with the driver. |

## Officially supported but metadata-only here

| Op | TFLite op | Sizes | Payload | Gap layer | Notes |
| --- | --- | --- | --- | --- | --- |
| `reshape` | `RESHAPE` | `8, 16, 32` | no | Metadata-only op | The standalone reshape case does not emit a vela_model block. On this fastpath it behaves like a zero-runtime metadata/view operation rather than a delegated NPU workload. |

## Officially supported but not delegated here

| Op | TFLite op | Sizes | Payload | Gap layer | Notes |
| --- | --- | --- | --- | --- | --- |
| `argmax` | `ARG_MAX` | `8, 16` | no | ExecuTorch lowering/partition gap | Even with valid depth-axis and last-dimension cases, the current fastpath export does not emit a vela_model block. The Arm backend carries the int64-to-int32 cleanup pass for argmax outputs, and the stable-diffusion partitioner tests still count argmax as a leftover op after partitioning, so there is no delegated payload on the i.MX93 Linux path today. |
| `prelu` | `PRELU` | `8, 16` | no | ExecuTorch lowering/partition gap | Current fastpath export does not emit a vela_model block for this case. There is no dedicated prelu lowering or operator-support wiring under backends/arm on this path, so device results are CPU-path only. |

## Officially supported but blocked at export

| Op | TFLite op | Sizes | Payload | Gap layer | Notes |
| --- | --- | --- | --- | --- | --- |
| `resize_bilinear` | `RESIZE_BILINEAR` | `8, 16` | no | Vela compile-time gap | Current i.MX93/U65 export fails inside Vela/regor before device execution. The failure reproduces across UpsamplingBilinear2d, Upsample(..., mode='bilinear', align_corners=True), and interpolate(..., mode='bilinear', align_corners=True) variants, while in-tree hardware tests only cover U85 delegation and explicitly mark U55 as not delegated. |

## Delegated but not inference-safe on i.MX93

| Op | TFLite op | Sizes | Payload | Gap layer | Notes |
| --- | --- | --- | --- | --- | --- |
| `strided_slice` | `STRIDED_SLICE` | `16` | yes | i.MX93 runtime gap | Delegates, but current device execution is not inference-safe on i.MX93. A simpler upstream-style 2D step-3 case completes and still returns obviously wrong values, while the earlier 4D case was numerically wrong and a simplified 3D single-axis case hangs the Linux fastpath before writing outputs. |
| `unpack` | `UNPACK` | `8, 16` | yes | i.MX93 runtime gap | Delegates, but the current i.MX93 Linux fastpath is not inference-safe for unpack. An upstream-style 3D dim-0 case fails in the NXP driver with IOCTL failure, while upstream-style and microbench 4D dim-2 cases execute and still return numerically wrong outputs even after checking output permutations. |

## Constraint Boundaries For The Open Delta Set

### `argmax`

- TFLite op: `ARG_MAX`
- Payload emitted: no
- Gap layer: ExecuTorch lowering/partition gap
- Model guidance: Keep argmax outside the delegated region on i.MX93.
- Fastpath outcome: Even with valid depth-axis and last-dimension cases, the current fastpath export does not emit a vela_model block. The Arm backend carries the int64-to-int32 cleanup pass for argmax outputs, and the stable-diffusion partitioner tests still count argmax as a leftover op after partitioning, so there is no delegated payload on the i.MX93 Linux path today.
- Official U55/U65 constraints:
  - IFM must be int8 or uint8.
  - OFM must be int32 or int64.
  - Reduction must run along the depth axis.
  - IFM depth must be <= 127.

### `prelu`

- TFLite op: `PRELU`
- Payload emitted: no
- Gap layer: ExecuTorch lowering/partition gap
- Model guidance: Do not rely on delegated prelu on i.MX93; use a different activation or allow CPU fallback.
- Fastpath outcome: Current fastpath export does not emit a vela_model block for this case. There is no dedicated prelu lowering or operator-support wiring under backends/arm on this path, so device results are CPU-path only.
- Official U55/U65 constraints:
  - All required operator attributes must be specified.
  - Input and output tensors must be static and have defined shapes.
  - Output tensors cannot be scalar except for QUANTIZE.
  - Input and output tensors must be at most 4D.
  - Tensors must use int16, int32, int8, or uint8 except for ARG_MAX outputs.
  - Input, output, and weight tensors must have quantization parameters except for ARG_MAX, MIRROR_PAD, SHAPE, and TRANSPOSE.
  - Tensor dimensions must stay in [1, 65535].
  - Per-axis quantization is only supported for CONV_2D, DEPTHWISE_CONV_2D, FULLY_CONNECTED, and TRANSPOSE_CONV.
  - IFM batch size must be 1 except for FULLY_CONNECTED, RESHAPE, SHAPE, SLICE, SOFTMAX, SPLIT, SPLIT_V, SQUEEZE, STRIDED_SLICE, and UNPACK.
  - Fused activations are limited to LOGISTIC, RELU, RELU6, RELU_0_TO_1, RELU_N1_TO_1, and TANH.

### `reshape`

- TFLite op: `RESHAPE`
- Payload emitted: no
- Gap layer: Metadata-only op
- Model guidance: Treat reshape as bookkeeping, not as a delegated compute op to benchmark.
- Fastpath outcome: The standalone reshape case does not emit a vela_model block. On this fastpath it behaves like a zero-runtime metadata/view operation rather than a delegated NPU workload.
- Official U55/U65 constraints:
  - Input and output quantization must match.
  - Input and output element counts must match.
  - Target shape must be constant.

### `resize_bilinear`

- TFLite op: `RESIZE_BILINEAR`
- Payload emitted: no
- Gap layer: Vela compile-time gap
- Model guidance: Avoid delegated bilinear resize on i.MX93 until the Vela U65 export path is resolved.
- Fastpath outcome: Current i.MX93/U65 export fails inside Vela/regor before device execution. The failure reproduces across UpsamplingBilinear2d, Upsample(..., mode='bilinear', align_corners=True), and interpolate(..., mode='bilinear', align_corners=True) variants, while in-tree hardware tests only cover U85 delegation and explicitly mark U55 as not delegated.
- Official U55/U65 constraints:
  - IFM/OFM width and height must either match, be 1, or scale by 1x/2x/4x/8x under the align_corners rules.
  - Size tensor must match the OFM shape.
  - align_corners and half_pixel_centers cannot both be True.
  - For half_pixel_centers, IFM width and height must be 1 or OFM must be exactly 2x IFM.

### `strided_slice`

- TFLite op: `STRIDED_SLICE`
- Payload emitted: yes
- Gap layer: i.MX93 runtime gap
- Model guidance: Avoid delegated strided_slice on i.MX93 until the runtime hang is understood.
- Fastpath outcome: Delegates, but current device execution is not inference-safe on i.MX93. A simpler upstream-style 2D step-3 case completes and still returns obviously wrong values, while the earlier 4D case was numerically wrong and a simplified 3D single-axis case hangs the Linux fastpath before writing outputs.
- Official U55/U65 constraints:
  - Exactly 4 input tensors are required.
  - Begin, end, and stride tensors must be constant.
  - ellipsis_mask must be 0.
  - new_axis_mask and shrink_axis_mask cannot both be set.
  - Slice end values must exceed begin values.
  - Batch and channel strides must be 1.
  - Offset attribute must be False.

### `unpack`

- TFLite op: `UNPACK`
- Payload emitted: yes
- Gap layer: i.MX93 runtime gap
- Model guidance: Do not rely on delegated unpack outputs on i.MX93 until the runtime numerical mismatch is resolved.
- Fastpath outcome: Delegates, but the current i.MX93 Linux fastpath is not inference-safe for unpack. An upstream-style 3D dim-0 case fails in the NXP driver with IOCTL failure, while upstream-style and microbench 4D dim-2 cases execute and still return numerically wrong outputs even after checking output permutations.
- Official U55/U65 constraints:
  - All required operator attributes must be specified.
  - Input and output tensors must be static and have defined shapes.
  - Output tensors cannot be scalar except for QUANTIZE.
  - Input and output tensors must be at most 4D.
  - Tensors must use int16, int32, int8, or uint8 except for ARG_MAX outputs.
  - Input, output, and weight tensors must have quantization parameters except for ARG_MAX, MIRROR_PAD, SHAPE, and TRANSPOSE.
  - Tensor dimensions must stay in [1, 65535].
  - Per-axis quantization is only supported for CONV_2D, DEPTHWISE_CONV_2D, FULLY_CONNECTED, and TRANSPOSE_CONV.
  - IFM batch size must be 1 except for FULLY_CONNECTED, RESHAPE, SHAPE, SLICE, SOFTMAX, SPLIT, SPLIT_V, SQUEEZE, STRIDED_SLICE, and UNPACK.
  - Fused activations are limited to LOGISTIC, RELU, RELU6, RELU_0_TO_1, RELU_N1_TO_1, and TANH.

## Delta Summary

| Category | Count |
| --- | ---: |
| Clean delegated on i.MX93 | 9 |
| Officially supported but metadata-only here | 1 |
| Officially supported but not delegated here | 2 |
| Officially supported but blocked at export | 1 |
| Delegated but not inference-safe on i.MX93 | 2 |

