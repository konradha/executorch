# Copyright 2023-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import os
import struct
import tempfile
from typing import List

import numpy as np

try:
    from ethosu.vela import vela  # type: ignore

    has_vela = True
except ImportError:
    has_vela = False

# Wire constants shared with VelaBinStream.h.
_VELA_BLOCK_ALIGNMENT = 16
_VELA_BLOCK_NAME_LENGTH = 16
# Twelve reserved bytes make the header 32 bytes.
_VELA_BLOCK_HEADER = struct.Struct("<16sI12x")
_VELA_IO_COUNT = struct.Struct("<i")
_VELA_IO = struct.Struct("<9i")
_VELA_IO_SHAPE_DIMENSIONS = 6

# ExecuTorch-only marker; never pass it to Vela.
NXP_VELA_MODEL_FLAG = "--embed-nxp-vela-model"


def _as_int32(value, name: str) -> int:
    """Convert a NumPy scalar to signed int32."""
    arr = np.asarray(value)
    if np.issubdtype(arr.dtype, np.unsignedinteger):
        # Preserve Vela's two's-complement sentinel values.
        arr = arr.astype(np.int64)
    v = int(arr)
    if v < -(2**31) or v > 2**31 - 1:
        raise ValueError(f"{name} out of int32 range: {v}")
    return v


def vela_bin_pack_io(prefix, data):
    vela_input_shapes = data[prefix + "_shape"]
    ios = bytearray(_VELA_IO_COUNT.pack(len(vela_input_shapes)))
    for i in range(len(vela_input_shapes)):
        io_shape = vela_input_shapes[i]
        io_elem_size = _as_int32(data[prefix + "_elem_size"][i], f"{prefix}_elem_size")
        io_offset = _as_int32(data[prefix + "_offset"][i], f"{prefix}_offset")
        io_region = _as_int32(data[prefix + "_region"][i], f"{prefix}_region")
        if len(io_shape) != _VELA_IO_SHAPE_DIMENSIONS:
            raise ValueError(
                f"Expected {_VELA_IO_SHAPE_DIMENSIONS}D shape, got {len(io_shape)}D"
            )
        ios.extend(
            _VELA_IO.pack(*io_shape.tolist(), io_elem_size, io_offset, io_region)
        )
    return bytes(ios)


def vela_compile(
    tosa_flatbuffer: bytes,
    args: List[str],
    verbose: bool = False,
    intermediate_path: str | None = None,
):
    """Compile a TOSA graph to an ArmBackendEthosU binary stream."""
    if not has_vela:
        raise RuntimeError(
            "ethos-u-vela pip package couldn't be imported. Make sure it's installed!"
        )

    compile_args = [arg for arg in args if arg != NXP_VELA_MODEL_FLAG]
    embed_nxp_model = len(compile_args) != len(args)

    def run(dir: str) -> bytes:
        tosaname = "out.tosa"
        tosa_path = os.path.join(dir, tosaname)
        with open(tosa_path, "wb") as f:
            f.write(tosa_flatbuffer)

        output_dir = os.path.join(dir, "output")
        raw_args = [
            *compile_args,
            f"--output-dir={output_dir}",
            tosa_path,
        ]
        if verbose:
            raw_args.append("--verbose-all")
        vela.main(raw_args)

        np_path = os.path.join(dir, "output", "out_vela.npz")

        blocks = bytearray()
        with np.load(np_path, allow_pickle=False) as data:
            # Convert Vela's NPZ output to the runtime block stream.
            bin_blocks = {"vela_bin_stream": b""}

            bin_blocks["cmd_data"] = data["cmd_data"].tobytes()

            bin_blocks["weight_data"] = data["weight_data"].tobytes()

            # scratch_shape stores the required scratch bytes.
            if not isinstance(data["scratch_shape"][0], np.int64):
                raise RuntimeError("Expected scratch to be int64")
            block_length = int(data["scratch_shape"][0])
            bin_blocks["scratch_size"] = struct.pack("<I", block_length)

            bin_blocks["inputs"] = vela_bin_pack_io("input", data)
            bin_blocks["outputs"] = vela_bin_pack_io("output", data)

            if embed_nxp_model:
                tflite_dir = os.path.join(dir, "output_tflite")
                tflite_args = [
                    arg
                    for arg in compile_args
                    if not arg.startswith("--output-format")
                    and not arg.startswith("--output-dir")
                ]
                tflite_args.extend(
                    [
                        "--output-format=tflite",
                        f"--output-dir={tflite_dir}",
                        tosa_path,
                    ]
                )
                vela.main(tflite_args)
                tflite_path = os.path.join(tflite_dir, "out_vela.tflite")
                if not os.path.isfile(tflite_path):
                    raise RuntimeError("Vela did not produce the NXP TFLite model.")
                with open(tflite_path, "rb") as tflite_file:
                    bin_blocks["vela_model"] = tflite_file.read()

            bin_blocks["vela_end_stream"] = b""

            for name, payload in bin_blocks.items():
                block_name = name.encode("utf8")[: _VELA_BLOCK_NAME_LENGTH - 1]
                blocks.extend(_VELA_BLOCK_HEADER.pack(block_name, len(payload)))
                blocks.extend(payload)
                blocks.extend(b"\x00" * (-len(payload) % _VELA_BLOCK_ALIGNMENT))

        return bytes(blocks)

    if intermediate_path is not None:
        return run(intermediate_path)
    else:
        with tempfile.TemporaryDirectory() as tmpdir:
            return run(tmpdir)
