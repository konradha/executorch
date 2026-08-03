#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import argparse
import json
import os
import subprocess  # nosec B404 - launches trusted local tooling
import sys
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_TARGET = "ethos-u65-256"
NXP_VELA_MODEL_FLAG = "--embed-nxp-vela-model"
DEFAULT_SYSTEM_CONFIG = "Ethos_U65_High_End"
DEFAULT_MEMORY_MODE = "Dedicated_Sram"
# The i.MX93 exposes 96 KiB of non-secure SRAM to the Ethos-U.
NPU_SRAM_BYTES = 96 * 1024
DEFAULT_COMPILER_FLAGS = (
    f"--arena-cache-size={NPU_SRAM_BYTES}",
    NXP_VELA_MODEL_FLAG,
)


def strip_export_guards(graph_module):
    for node in list(graph_module.graph.nodes):
        if node.op == "call_module" and (
            node.target == "_guards_fn" or node.name == "_guards_fn"
        ):
            graph_module.graph.erase_node(node)

    if hasattr(graph_module, "_guards_fn"):
        delattr(graph_module, "_guards_fn")

    graph_module.graph.lint()
    graph_module.recompile()
    return graph_module


def build_export_command(
    model_name: str,
    output_dir: str | Path,
    *,
    target: str = DEFAULT_TARGET,
    system_config: str = DEFAULT_SYSTEM_CONFIG,
    memory_mode: str = DEFAULT_MEMORY_MODE,
    extra_compiler_flags: Iterable[str] = DEFAULT_COMPILER_FLAGS,
    channels_last_4d: bool = False,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "examples.arm.aot_arm_compiler",
        "-m",
        model_name,
        "-t",
        target,
        "-q",
        "-d",
        "-o",
        str(output_dir),
        "--system_config",
        system_config,
        "--memory_mode",
        memory_mode,
    ]
    for flag in extra_compiler_flags:
        command.append(f"--extra_compiler_flag={flag}")
    if channels_last_4d:
        command.append("--channels_last_4d")
    quantized_ops_library = detect_quantized_ops_library()
    if quantized_ops_library is not None:
        command.extend(["-s", quantized_ops_library])
    return command


def build_export_env(base_env: dict[str, str] | None = None) -> dict[str, str]:
    return dict(base_env or {})


def detect_quantized_ops_library() -> str | None:
    override = os.environ.get("EXECUTORCH_QUANTIZED_OPS_LIBRARY")
    if override:
        return override
    try:
        from executorch.extension.pybindings import portable_lib  # noqa: F401
    except Exception:
        return None
    module_path = Path(portable_lib.__file__).resolve()
    package_root = module_path.parent
    for parent in module_path.parents:
        if parent.name == "executorch":
            package_root = parent
            break

    # Search only the loaded package and this checkout. Walking every ancestor
    # can otherwise turn a missing optional library into a scan of the host.
    search_roots = dict.fromkeys((module_path.parent, package_root, REPO_ROOT))
    for root in search_roots:
        for match in sorted(root.glob("**/*quantized_ops_aot_lib.*")):
            if match.is_file():
                return str(match)
    return None


def export_model(
    model_name: str,
    output_dir: str | Path,
    *,
    target: str = DEFAULT_TARGET,
    system_config: str = DEFAULT_SYSTEM_CONFIG,
    memory_mode: str = DEFAULT_MEMORY_MODE,
    extra_compiler_flags: Iterable[str] = DEFAULT_COMPILER_FLAGS,
    channels_last_4d: bool = False,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    command = build_export_command(
        model_name,
        output_dir,
        target=target,
        system_config=system_config,
        memory_mode=memory_mode,
        extra_compiler_flags=extra_compiler_flags,
        channels_last_4d=channels_last_4d,
    )
    subprocess.run(  # nosec B603 - command assembled from vetted local values
        command,
        check=True,
        cwd=REPO_ROOT,
        env=build_export_env(os.environ),
    )

    pte_files = sorted(output_dir.glob("*.pte"))
    if not pte_files:
        raise FileNotFoundError(f"No .pte file found in {output_dir}")
    return max(pte_files, key=lambda candidate: candidate.stat().st_size)


def verify_device_output(reference_path: str | Path, device_argmax: int) -> bool:
    with open(reference_path) as reference_file:
        reference = json.load(reference_file)

    host_argmax = reference["argmax"]
    top5 = reference["top5_indices"]
    return device_argmax == host_argmax or device_argmax in top5


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", help="Model name understood by aot_arm_compiler")
    parser.add_argument("--output", default="arm_test/models/imx93")
    parser.add_argument("--target", default=DEFAULT_TARGET)
    parser.add_argument("--system_config", default=DEFAULT_SYSTEM_CONFIG)
    parser.add_argument("--memory_mode", default=DEFAULT_MEMORY_MODE)
    parser.add_argument(
        "--extra_compiler_flag",
        action="append",
        default=list(DEFAULT_COMPILER_FLAGS),
        help="Additional Vela flags. Repeat for multiple values.",
    )
    parser.add_argument("--verify", help="Path to a JSON reference file")
    parser.add_argument("--device_argmax", type=int)
    args = parser.parse_args()

    if args.verify is not None:
        if args.device_argmax is None:
            raise SystemExit("--device_argmax is required with --verify")
        raise SystemExit(
            0 if verify_device_output(args.verify, args.device_argmax) else 1
        )

    if args.model is None:
        raise SystemExit("--model is required unless --verify is used")

    pte_path = export_model(
        args.model,
        args.output,
        target=args.target,
        system_config=args.system_config,
        memory_mode=args.memory_mode,
        extra_compiler_flags=args.extra_compiler_flag,
    )
    print(pte_path)


if __name__ == "__main__":
    main()
