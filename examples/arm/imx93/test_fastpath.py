# SPDX-License-Identifier: BSD-3-Clause

import unittest

from pathlib import Path
from unittest import mock

import torch

import numpy as np
import tempfile

from backends.arm import arm_vela
from backends.arm.scripts import aot_arm_compiler
from examples.arm.imx93.operator_benchmarks import (
    BENCH_OPS,
    build_scp_command,
    build_ssh_command,
    parse_export_output,
    parse_runner_output,
    trim_output,
)
from examples.arm.imx93.export_and_verify import (
    DEFAULT_COMPILER_FLAGS,
    DEFAULT_MEMORY_MODE,
    DEFAULT_SYSTEM_CONFIG,
    build_export_command,
    build_export_env,
    detect_quantized_ops_library,
    strip_export_guards,
    verify_device_output,
)
from examples.arm.imx93.operator_microbench_model import build_case, execution_inputs
from examples.arm.imx93.operator_sweep import (
    _comparison_tensor,
    _load_device_tensor,
    _reference_delta,
    _save_tensor_sequence,
    _tensor_storage_bytes,
)
from examples.arm.imx93.fastpath_delta_report import render_markdown
from examples.arm.imx93.model_benchmarks import (
    MODEL_PARAMS_MILLIONS,
    summarize_device_output,
)
from examples.arm.imx93.supported_ops import (
    IMX93_FASTPATH_MATRIX,
    MODEL_PROVEN_CORE_OPS,
    summarize_ops,
    trusted_delegated_ops,
    unresolved_ops,
)


class _IdentityModule(torch.nn.Module):
    def forward(self, x):
        return x


class FastpathUtilsTest(unittest.TestCase):
    def test_build_export_command_uses_imx_defaults(self):
        command = build_export_command("mv2", "arm_test/models/imx93")
        self.assertEqual(DEFAULT_SYSTEM_CONFIG, "Ethos_U65_High_End")
        self.assertIn("--system_config", command)
        self.assertIn(DEFAULT_SYSTEM_CONFIG, command)
        self.assertIn("--memory_mode", command)
        self.assertIn(DEFAULT_MEMORY_MODE, command)
        self.assertNotIn("--channels_last_4d", command)
        for flag in DEFAULT_COMPILER_FLAGS:
            self.assertIn(f"--extra_compiler_flag={flag}", command)

    def test_build_export_command_can_enable_channels_last(self):
        command = build_export_command(
            "mv2",
            "arm_test/models/imx93",
            channels_last_4d=True,
        )
        self.assertIn("--channels_last_4d", command)

    def test_prepare_model_and_inputs_for_export_uses_channels_last_for_rank4(self):
        module = _IdentityModule().eval()
        tensor = torch.randn(1, 8, 16, 16)
        prepared_module, prepared_inputs = (
            aot_arm_compiler.prepare_model_and_inputs_for_export(
                module,
                (tensor,),
                True,
            )
        )
        self.assertTrue(next(prepared_module.parameters(), torch.empty(0)).numel() == 0)
        self.assertTrue(
            prepared_inputs[0].is_contiguous(memory_format=torch.channels_last)
        )

    def test_build_export_env_prefers_local_checkout(self):
        env = build_export_env({"PYTHONPATH": "/tmp/site"})
        self.assertEqual(env["PYTHONPATH"], "/tmp/site")

    def test_detect_quantized_ops_library_prefers_env_override(self):
        with mock.patch.dict(
            "os.environ",
            {"EXECUTORCH_QUANTIZED_OPS_LIBRARY": "/tmp/libquantized_ops_aot_lib.dylib"},
            clear=False,
        ):
            self.assertEqual(
                detect_quantized_ops_library(),
                "/tmp/libquantized_ops_aot_lib.dylib",
            )

    def test_strip_export_guards_removes_guard_module(self):
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        graph.call_module("_guards_fn", (x,))
        graph.output(x)
        module = torch.fx.GraphModule(
            {"_guards_fn": _IdentityModule()},
            graph,
        )

        stripped = strip_export_guards(module)

        self.assertFalse(hasattr(stripped, "_guards_fn"))
        self.assertFalse(
            any(node.target == "_guards_fn" for node in stripped.graph.nodes)
        )

    def test_verify_device_output_accepts_top5_match(self):
        import json
        import tempfile

        payload = {"argmax": 4, "top5_indices": [4, 9, 3, 2, 1]}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json") as handle:
            json.dump(payload, handle)
            handle.flush()
            self.assertTrue(verify_device_output(handle.name, 9))
            self.assertFalse(verify_device_output(handle.name, 99))

    def test_supported_ops_summary_is_non_empty(self):
        summary = summarize_ops()
        self.assertGreater(summary["int_ops"], 0)
        self.assertGreater(summary["fp_extra_ops"], 0)
        self.assertGreater(summary["models"], 0)
        self.assertGreater(summary["model_proven_core_ops"], 0)
        self.assertGreater(summary["under_documented_tflite_ops"], 0)
        self.assertGreater(summary["benchmark_hints"], 0)
        self.assertGreater(summary["fastpath_matrix"], 0)
        self.assertGreater(summary["trusted_delegated"], 0)
        self.assertGreater(summary["unresolved_fastpath"], 0)

    def test_supported_ops_matrix_tracks_device_outcomes(self):
        self.assertEqual(
            IMX93_FASTPATH_MATRIX["argmax"]["observed_status"], "not_delegated"
        )
        self.assertIn(
            IMX93_FASTPATH_MATRIX["depthwise_conv2d"]["observed_status"],
            {"correct", "correctness_issue"},
        )
        self.assertEqual(
            IMX93_FASTPATH_MATRIX["logistic"]["observed_status"],
            "correct",
        )
        self.assertEqual(IMX93_FASTPATH_MATRIX["cat"]["observed_status"], "correct")
        self.assertEqual(
            IMX93_FASTPATH_MATRIX["resize_bilinear"]["observed_status"],
            "export_failed",
        )
        self.assertEqual(
            IMX93_FASTPATH_MATRIX["resize_nearest_neighbor"]["observed_status"],
            "correct",
        )
        self.assertEqual(
            IMX93_FASTPATH_MATRIX["strided_slice"]["observed_status"],
            "unsafe_for_inference",
        )
        self.assertEqual(
            IMX93_FASTPATH_MATRIX["unpack"]["observed_status"],
            "unsafe_for_inference",
        )
        self.assertEqual(
            IMX93_FASTPATH_MATRIX["strided_slice"]["gap_layer"],
            "imx_runtime",
        )
        self.assertFalse(IMX93_FASTPATH_MATRIX["argmax"]["delegated_payload"])
        self.assertTrue(IMX93_FASTPATH_MATRIX["resize_nearest_neighbor"]["delegated_payload"])

    def test_supported_ops_trusted_and_unresolved_sets_are_stable(self):
        trusted = trusted_delegated_ops()
        unresolved = unresolved_ops()
        self.assertIn("aten_convolution_default", MODEL_PROVEN_CORE_OPS)
        self.assertIn("aten_add_tensor", MODEL_PROVEN_CORE_OPS)
        self.assertIn("depthwise_conv2d", trusted)
        self.assertIn("resize_nearest_neighbor", trusted)
        self.assertNotIn("argmax", trusted)
        self.assertIn("argmax", unresolved)
        self.assertIn("resize_bilinear", unresolved)
        self.assertIn("unpack", unresolved)

    def test_fastpath_delta_report_mentions_gap_layers(self):
        report = render_markdown()
        self.assertIn("## Trusted Delegated Set", report)
        self.assertIn("## Model-Proven Core Path", report)
        self.assertIn("## Constraint Boundaries For The Open Delta Set", report)
        self.assertIn("ExecuTorch lowering/partition gap", report)
        self.assertIn("Vela compile-time gap", report)
        self.assertIn("i.MX93 runtime gap", report)
        self.assertIn("Delegated but not inference-safe on i.MX93", report)
        self.assertIn("`depthwise_conv2d`", report)
        self.assertIn("`resize_bilinear`", report)

    def test_operator_benchmark_parser_extracts_metrics(self):
        payload = (
            "Ethos-U i.MX cycle counter: 12345\n"
            "Model executed successfully 10 time(s) in 4.250000 ms.\n"
        )
        metrics = parse_runner_output(payload)
        self.assertEqual(metrics["cycle_counter"], 12345)
        self.assertEqual(metrics["executions"], 10)
        self.assertEqual(metrics["total_ms"], 4.25)

    def test_operator_benchmark_list_is_non_empty(self):
        self.assertIn("depthwise_conv2d", BENCH_OPS)

    def test_operator_export_parser_extracts_estimates(self):
        payload = "NPU operators = 5\nTotal cycles                                      3984 cycles/batch\n"
        metrics = parse_export_output(payload)
        self.assertEqual(metrics["npu_operators"], 5)
        self.assertEqual(metrics["estimated_cycles"], 3984)

    def test_operator_microbench_build_case_respects_size(self):
        _, argmax_inputs = build_case("argmax", 16)
        self.assertEqual(argmax_inputs[0].shape, (1, 8, 16, 16))
        _, example_inputs = build_case("depthwise_conv2d", 32)
        self.assertEqual(example_inputs[0].shape, (1, 8, 32, 32))
        _, reshape_inputs = build_case("reshape", 16)
        self.assertEqual(reshape_inputs[0].shape, (16, 16))
        _, strided_slice_inputs = build_case("strided_slice", 16)
        self.assertEqual(strided_slice_inputs[0].shape, (2, 16, 4))
        _, unpack_inputs = build_case("unpack", 16)
        self.assertEqual(unpack_inputs[0].shape, (1, 4, 4, 16))

    def test_operator_microbench_execution_inputs_are_ones_like(self):
        _, example_inputs = build_case("cat", 8)
        inputs = execution_inputs(example_inputs)
        self.assertEqual(len(inputs), 2)
        self.assertTrue(torch.equal(inputs[0], example_inputs[0]))
        self.assertTrue(torch.equal(inputs[1], example_inputs[1]))
        self.assertIsNot(inputs[0], example_inputs[0])
        self.assertIsNot(inputs[1], example_inputs[1])

    def test_operator_microbench_depthwise_conv_is_identity_initialized(self):
        module, _ = build_case("depthwise_conv2d", 8)
        weight = module.conv.weight.detach()
        bias = module.conv.bias.detach()
        self.assertTrue(torch.allclose(weight[:, :, 1, 1], torch.ones(8, 1)))
        self.assertEqual(float(weight.sum().item()), 8.0)
        self.assertTrue(torch.equal(bias, torch.zeros_like(bias)))

    def test_operator_sweep_load_device_tensor_accepts_narrower_int_width(self):
        host = torch.arange(8, dtype=torch.int64).reshape(1, 8)
        with tempfile.NamedTemporaryFile(suffix=".bin") as handle:
            np.arange(8, dtype=np.int32).tofile(handle.name)
            device = _load_device_tensor(Path(handle.name), host)
        self.assertEqual(device.dtype, np.int32)
        self.assertTrue(np.array_equal(device, host.numpy().astype(np.int32)))

    def test_operator_sweep_load_device_tensor_respects_channels_last_storage(self):
        host = torch.arange(32, dtype=torch.float32).reshape(1, 2, 4, 4)
        host = host.contiguous(memory_format=torch.channels_last)
        with tempfile.NamedTemporaryFile(suffix=".bin") as handle:
            Path(handle.name).write_bytes(_tensor_storage_bytes(host))
            device = _load_device_tensor(Path(handle.name), host)
        self.assertTrue(np.array_equal(device, host.numpy()))

    def test_operator_sweep_comparison_tensor_makes_views_contiguous(self):
        view = torch.arange(16, dtype=torch.float32).reshape(4, 4)[:, ::2]
        compared = _comparison_tensor(view)
        self.assertTrue(compared.is_contiguous())
        self.assertTrue(torch.equal(compared, view.contiguous()))

    def test_operator_sweep_reference_delta_is_zero_for_identical_outputs(self):
        tensor = torch.arange(8, dtype=torch.float32)
        metrics = _reference_delta((tensor,), (tensor.clone(),))
        self.assertEqual(metrics["max_abs_error"], 0.0)
        self.assertEqual(metrics["rmse"], 0.0)

    def test_model_benchmark_summary_extracts_topk_matches(self):
        output = np.array([[0.1, 0.9, 0.2, 0.3, 0.4]], dtype=np.float32)
        reference_output = np.array([[0.1, 0.9, 0.2, 0.3, 0.4]], dtype=np.float32)
        summary = summarize_device_output(output, reference_output)
        self.assertEqual(summary["device_argmax"], 1)
        self.assertEqual(summary["reference_argmax"], 1)
        self.assertEqual(summary["top1_match"], 1)
        self.assertEqual(summary["top5_match"], 1)
        self.assertLess(summary["max_abs_error"], 1e-6)
        self.assertLess(summary["rmse"], 1e-6)
        self.assertLess(summary["first10_max_abs_error"], 1e-6)
        self.assertLess(summary["first10_rmse"], 1e-6)

    def test_model_benchmark_params_cover_small_and_large_models(self):
        self.assertLess(MODEL_PARAMS_MILLIONS["mv3"], MODEL_PARAMS_MILLIONS["resnet50"])
        self.assertLess(
            MODEL_PARAMS_MILLIONS["resnet50"],
            MODEL_PARAMS_MILLIONS["large_convnet"],
        )

    def test_operator_sweep_save_tensor_sequence_writes_npy_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            _save_tensor_sequence((torch.arange(4, dtype=torch.float32),), output_dir, "reference")
            saved = np.load(output_dir / "reference-0.npy")
        self.assertTrue(np.array_equal(saved, np.arange(4, dtype=np.float32)))

    def test_operator_export_trim_keeps_tail(self):
        payload = "\n".join(f"line-{idx}" for idx in range(30))
        self.assertEqual(trim_output(payload, line_count=3), "line-27\nline-28\nline-29")

    def test_ssh_and_scp_commands_accept_custom_options(self):
        ssh_command = build_ssh_command(
            "user@host",
            "echo ok",
            ["ProxyCommand=example proxy", "StrictHostKeyChecking=no"],
        )
        scp_command = build_scp_command(
            "/tmp/local.bin",
            "user@host",
            "/tmp/remote.bin",
            ["ProxyCommand=example proxy"],
        )
        self.assertEqual(
            ssh_command[:5],
            ["ssh", "-o", "ProxyCommand=example proxy", "-o", "StrictHostKeyChecking=no"],
        )
        self.assertEqual(ssh_command[-2:], ["user@host", "echo ok"])
        self.assertEqual(
            scp_command,
            [
                "scp",
                "-o",
                "ProxyCommand=example proxy",
                "/tmp/local.bin",
                "user@host:/tmp/remote.bin",
            ],
        )

    def test_vela_compile_includes_vela_model_block(self):
        with mock.patch.object(arm_vela, "has_vela", True):
            with mock.patch.object(arm_vela, "vela", create=True) as fake_vela:
                def fake_main(argv):
                    output_dir = None
                    for item in argv:
                        if item.startswith("--output-dir="):
                            output_dir = item.split("=", 1)[1]
                    self.assertIsNotNone(output_dir)
                    Path(output_dir).mkdir(parents=True, exist_ok=True)
                    if any(item == "--output-format=tflite" for item in argv):
                        Path(output_dir, "out_vela.tflite").write_bytes(b"imx-tflite")
                        return
                    np.savez(
                        Path(output_dir, "out_vela.npz"),
                        cmd_data=np.array([1, 2], dtype=np.uint8),
                        weight_data=np.array([3, 4], dtype=np.uint8),
                        scratch_shape=np.array([64], dtype=np.int64),
                        input_shape=np.array([[1, 1, 1, 1, 1, 1]], dtype=np.int32),
                        input_elem_size=np.array([1], dtype=np.int32),
                        input_offset=np.array([0], dtype=np.int32),
                        input_region=np.array([0], dtype=np.int32),
                        output_shape=np.array([[1, 1, 1, 1, 1, 1]], dtype=np.int32),
                        output_elem_size=np.array([1], dtype=np.int32),
                        output_offset=np.array([0], dtype=np.int32),
                        output_region=np.array([0], dtype=np.int32),
                    )

                fake_vela.main.side_effect = fake_main
                payload = arm_vela.vela_compile(b"tosa", ["--output-format=raw"])

        self.assertIn(b"vela_model", payload)
        self.assertIn(b"imx-tflite", payload)


if __name__ == "__main__":
    unittest.main()
