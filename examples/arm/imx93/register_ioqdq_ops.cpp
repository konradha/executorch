/*
 * Copyright 2026 Arm Limited and/or its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * Minimal kernel registration for quantize/dequantize ops.
 * Registers only the two IOQDQ ops needed for Ethos-U delegate wrappers,
 * avoiding the full cortex_m_ops_lib which overflows ITCM on 128KB targets.
 */

#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/kernel/operator_registry.h>

namespace cortex_m {
namespace native {
executorch::aten::Tensor& quantize_per_tensor_out(
    torch::executor::KernelRuntimeContext& context,
    const executorch::aten::Tensor& input,
    double scale,
    int64_t zero_point,
    int64_t quant_min,
    int64_t quant_max,
    executorch::aten::ScalarType dtype,
    executorch::aten::Tensor& out);

executorch::aten::Tensor& dequantize_per_tensor_out(
    torch::executor::KernelRuntimeContext& context,
    const executorch::aten::Tensor& input,
    double scale,
    int64_t zero_point,
    int64_t quant_min,
    int64_t quant_max,
    executorch::aten::ScalarType dtype,
    executorch::aten::Tensor& out);
} // namespace native
} // namespace cortex_m

namespace {
using executorch::runtime::EValue;
using executorch::runtime::Kernel;
using executorch::runtime::Span;

static Kernel kernels[] = {
    Kernel(
        "cortex_m::quantize_per_tensor.out",
        [](torch::executor::KernelRuntimeContext& context,
           Span<EValue*> stack) {
          cortex_m::native::quantize_per_tensor_out(
              context,
              stack[0]->to<executorch::aten::Tensor>(),
              stack[1]->to<double>(),
              stack[2]->to<int64_t>(),
              stack[3]->to<int64_t>(),
              stack[4]->to<int64_t>(),
              stack[5]->to<executorch::aten::ScalarType>(),
              stack[6]->to<executorch::aten::Tensor>());
        }),
    Kernel(
        "cortex_m::dequantize_per_tensor.out",
        [](torch::executor::KernelRuntimeContext& context,
           Span<EValue*> stack) {
          cortex_m::native::dequantize_per_tensor_out(
              context,
              stack[0]->to<executorch::aten::Tensor>(),
              stack[1]->to<double>(),
              stack[2]->to<int64_t>(),
              stack[3]->to<int64_t>(),
              stack[4]->to<int64_t>(),
              stack[5]->to<executorch::aten::ScalarType>(),
              stack[6]->to<executorch::aten::Tensor>());
        }),
};

static Span<const Kernel> kernel_span(kernels, kernels + 2);
static auto reg = executorch::runtime::register_kernels(kernel_span);
} // namespace
