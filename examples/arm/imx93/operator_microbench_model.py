#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os

import torch
import torch.nn.functional as F


class ArgMaxModule(torch.nn.Module):
    def forward(self, x):
        return torch.argmax(x, dim=1)


class CatModule(torch.nn.Module):
    def forward(self, a, b):
        return torch.cat((a, b), dim=1)


class DepthwiseConvModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(8, 8, kernel_size=3, padding=1, groups=8)
        with torch.no_grad():
            self.conv.weight.zero_()
            self.conv.weight[:, :, 1, 1] = 1.0
            self.conv.bias.zero_()

    def forward(self, x):
        return self.conv(x)


class LogisticModule(torch.nn.Module):
    def forward(self, x):
        return torch.sigmoid(x)


class PadModule(torch.nn.Module):
    def forward(self, x):
        return F.pad(x, (1, 1, 2, 2))


class PReLUModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.prelu = torch.nn.PReLU()
        with torch.no_grad():
            self.prelu.weight.fill_(0.25)

    def forward(self, x):
        return self.prelu(x)


class ReshapeModule(torch.nn.Module):
    def __init__(self, size: int):
        super().__init__()
        self.size = size

    def forward(self, x):
        return x.reshape(self.size * self.size)


class ResizeBilinearModule(torch.nn.Module):
    def __init__(self, size: int):
        super().__init__()
        self.upsample = torch.nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x):
        return self.upsample(x)


class ResizeNearestModule(torch.nn.Module):
    def __init__(self, size: int):
        super().__init__()
        self.size = size

    def forward(self, x):
        return F.interpolate(x, size=(self.size * 2, self.size * 2), mode="nearest")


class SliceModule(torch.nn.Module):
    def forward(self, x):
        return x[:, :, 2:-2, 1:-3]


class SplitModule(torch.nn.Module):
    def forward(self, x):
        return torch.split(x, 4, dim=1)


class SqueezeModule(torch.nn.Module):
    def forward(self, x):
        return x.squeeze(2)


class StridedSliceModule(torch.nn.Module):
    def __init__(self, dim: int, end: int):
        super().__init__()
        self.dim = dim
        self.end = end

    def forward(self, x):
        return torch.ops.aten.slice.Tensor(x, self.dim, 0, self.end, 2)


class TransposeModule(torch.nn.Module):
    def forward(self, x):
        return x.transpose(1, 3)


class UnpackModule(torch.nn.Module):
    def forward(self, x):
        return torch.unbind(x, dim=2)


def _pattern(shape, *, start: float = -1.0, end: float = 1.0):
    tensor = torch.linspace(start, end, steps=int(torch.tensor(shape).prod().item()))
    return tensor.reshape(shape)


def _single_input(size: int):
    return (_pattern((1, 8, size, size), start=0.0, end=1.0),)


def _pair_input(size: int):
    return (
        _pattern((1, 4, size, size), start=-1.0, end=0.0),
        _pattern((1, 4, size, size), start=0.25, end=1.25),
    )


def _with_seed(factory):
    with torch.random.fork_rng():
        torch.manual_seed(0)
        return factory()


def build_case(case_name: str, size: int):
    if case_name == "argmax":
        return ArgMaxModule(), (_pattern((1, 8, size, size), start=-2.0, end=2.0),)
    if case_name == "cat":
        return CatModule(), _pair_input(size)
    if case_name == "depthwise_conv2d":
        return _with_seed(DepthwiseConvModule), _single_input(size)
    if case_name == "logistic":
        return LogisticModule(), (_pattern((1, 8, size, size), start=-6.0, end=6.0),)
    if case_name == "pad":
        return PadModule(), _single_input(size)
    if case_name == "prelu":
        return _with_seed(PReLUModule), (_pattern((1, 6, size, size), start=-2.0, end=2.0),)
    if case_name == "reshape":
        return ReshapeModule(size), (_pattern((size, size), start=-1.0, end=1.0),)
    if case_name == "resize_bilinear":
        return ResizeBilinearModule(size), _single_input(size)
    if case_name == "resize_nearest_neighbor":
        return ResizeNearestModule(size), _single_input(size)
    if case_name == "slice":
        return SliceModule(), _single_input(size)
    if case_name == "split":
        return SplitModule(), _single_input(size)
    if case_name == "squeeze":
        return SqueezeModule(), (_pattern((1, 8, 1, size), start=-1.0, end=1.0),)
    if case_name == "strided_slice":
        return (
            StridedSliceModule(1, size),
            (_pattern((2, size, 4), start=0.0, end=1.0),),
        )
    if case_name == "transpose":
        return TransposeModule(), _single_input(size)
    if case_name == "unpack":
        return UnpackModule(), (_pattern((1, 4, 4, size), start=-1.0, end=1.0),)
    raise RuntimeError(f"Unknown IMX93_FASTPATH_BENCH_OP={case_name}")


def execution_inputs(example_inputs):
    return tuple(tensor.detach().clone() for tensor in example_inputs)


CASES = {
    "argmax",
    "cat",
    "depthwise_conv2d",
    "logistic",
    "pad",
    "prelu",
    "reshape",
    "resize_bilinear",
    "resize_nearest_neighbor",
    "slice",
    "split",
    "squeeze",
    "strided_slice",
    "transpose",
    "unpack",
}


_case_name = os.environ.get("IMX93_FASTPATH_BENCH_OP", "argmax")
if _case_name not in CASES:
    raise RuntimeError(f"Unknown IMX93_FASTPATH_BENCH_OP={_case_name}")
_case_size = int(os.environ.get("IMX93_FASTPATH_BENCH_SIZE", "16"))
if _case_size < 4:
    raise RuntimeError("IMX93_FASTPATH_BENCH_SIZE must be >= 4")

torch.manual_seed(0)
ModelUnderTest, _ExampleInputs = build_case(_case_name, _case_size)
ModelInputs = execution_inputs(_ExampleInputs)
