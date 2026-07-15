from __future__ import annotations

import copy
from pathlib import Path
import sys
import unittest

import torch
from transformers import GPT2Config, GPT2LMHeadModel


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from modules.torch_bpla import (
    AttentionDiagnostics,
    SharedBPLATables,
    TorchBPLAActivation,
    TorchBPLAConfig,
    TorchBPLALayerNorm,
    TorchBPLALinear,
    bpla_matmul_torch,
    bpla_multiply_torch,
    bpla_softmax_torch,
    calibrate_model_activation_range,
    replace_attention_matmuls,
    replace_layer_norms,
    replace_linear_and_gelu,
)


class TorchBPLAProxyTests(unittest.TestCase):
    def test_torch_multiply_proxy_runs(self):
        cfg = TorchBPLAConfig(prefix_bits=4, affine_path="dyadic", dyadic_terms=2)
        a = torch.randn(128)
        b = torch.randn(128)
        out = bpla_multiply_torch(a, b, cfg)
        self.assertEqual(out.shape, a.shape)
        self.assertTrue(torch.isfinite(out).all().item())

    def test_torch_activation_proxy_runs(self):
        cfg = TorchBPLAConfig(prefix_bits=4, affine_path="dyadic", dyadic_terms=2)
        x = torch.linspace(-4.0, 4.0, 256)
        out = TorchBPLAActivation("gelu", cfg)(x)
        self.assertEqual(out.shape, x.shape)
        self.assertTrue(torch.isfinite(out).all().item())

    def test_torch_linear_proxy_runs(self):
        cfg = TorchBPLAConfig(prefix_bits=3, affine_path="float", linear_chunk_out=4)
        layer = torch.nn.Linear(8, 6)
        out = TorchBPLALinear(layer, cfg)(torch.randn(5, 8))
        self.assertEqual(out.shape, (5, 6))
        self.assertTrue(torch.isfinite(out).all().item())

    def test_torch_batched_matmul_proxy_runs(self):
        cfg = TorchBPLAConfig(prefix_bits=3, affine_path="dyadic", dyadic_terms=2, linear_chunk_out=2)
        a = torch.randn(2, 3, 4, 5)
        b = torch.randn(2, 3, 5, 6)
        out = bpla_matmul_torch(a, b, cfg)
        self.assertEqual(out.shape, (2, 3, 4, 6))
        self.assertTrue(torch.isfinite(out).all().item())

    def test_composed_bpla_softmax_is_normalized_and_close(self):
        torch.manual_seed(0)
        cfg = TorchBPLAConfig(prefix_bits=4, affine_path="float")
        x = torch.randn(8, 17)
        actual = bpla_softmax_torch(x, dim=-1, config=cfg)
        expected = torch.softmax(x, dim=-1)
        self.assertTrue(torch.isfinite(actual).all().item())
        self.assertTrue((actual >= 0).all().item())
        self.assertLess((actual.sum(dim=-1) - 1.0).abs().max().item(), 1.0e-2)
        self.assertLess((actual - expected).abs().mean().item(), 1.0e-3)

    def test_composed_bpla_softmax_zeros_large_negative_masks(self):
        cfg = TorchBPLAConfig(prefix_bits=4, affine_path="float")
        x = torch.tensor([[1.0, 0.0, -1.0e9]])
        actual = bpla_softmax_torch(x, dim=-1, config=cfg)
        self.assertEqual(actual[0, 2].item(), 0.0)

    def test_composed_bpla_softmax_dyadic_path_remains_bounded(self):
        torch.manual_seed(2)
        cfg = TorchBPLAConfig(prefix_bits=4, affine_path="dyadic", dyadic_terms=2)
        actual = bpla_softmax_torch(torch.randn(8, 17), dim=-1, config=cfg)
        self.assertTrue(torch.isfinite(actual).all().item())
        self.assertTrue((actual >= 0).all().item())
        # Two signed-power-of-two terms are intentionally coarse; this guards
        # catastrophic divergence without presenting the setting as accurate.
        self.assertLess((actual.sum(dim=-1) - 1.0).abs().max().item(), 0.4)

    def test_composed_bpla_layernorm_is_close(self):
        torch.manual_seed(1)
        cfg = TorchBPLAConfig(prefix_bits=4, affine_path="float")
        source = torch.nn.LayerNorm(17)
        x = torch.randn(8, 17)
        actual = TorchBPLALayerNorm(source, cfg)(x)
        expected = source(x)
        self.assertTrue(torch.isfinite(actual).all().item())
        self.assertLess((actual - expected).abs().mean().item(), 2.0e-2)

    def test_replace_layer_norms_shares_tables(self):
        model = torch.nn.Sequential(torch.nn.LayerNorm(4), torch.nn.Sequential(torch.nn.LayerNorm(4)))
        cfg = TorchBPLAConfig(prefix_bits=3, affine_path="float")
        tables = SharedBPLATables(cfg)
        replaced = replace_layer_norms(model, cfg, tables)
        self.assertEqual(replaced, 2)
        self.assertIsInstance(model[0], TorchBPLALayerNorm)
        self.assertIsInstance(model[1][0], TorchBPLALayerNorm)
        self.assertIs(model[0].tables, model[1][0].tables)

    def test_converted_modules_share_multiplier_and_activation_tables(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = torch.nn.Linear(4, 4)
                self.gelu1 = torch.nn.GELU()
                self.fc2 = torch.nn.Linear(4, 4)
                self.gelu2 = torch.nn.GELU()

            def forward(self, x):
                return self.gelu2(self.fc2(self.gelu1(self.fc1(x))))

        cfg = TorchBPLAConfig(prefix_bits=3, affine_path="dyadic", dyadic_terms=2)
        tables = SharedBPLATables(cfg)
        model = Model()
        replace_linear_and_gelu(model, cfg, tables=tables)
        model(torch.randn(2, 4))
        self.assertIs(model.fc1.tables, model.fc2.tables)
        self.assertIs(model.gelu1.tables, model.gelu2.tables)
        self.assertIs(model.fc1.tables, model.gelu1.tables)
        self.assertEqual(len(tables._multiplier), 1)
        self.assertEqual(len(tables._activation), 1)

    def test_global_activation_calibration_observes_all_gelus(self):
        model = torch.nn.Sequential(torch.nn.GELU(), torch.nn.GELU())
        batches = [torch.tensor([[-1.0, 3.5]])]
        measured = calibrate_model_activation_range(model, batches, lambda m, x: m(x), max_batches=1)
        self.assertAlmostEqual(measured, 3.5)

    def test_exact_attention_callback_matches_gpt2(self):
        model = GPT2LMHeadModel(
            GPT2Config(vocab_size=64, n_positions=16, n_ctx=16, n_embd=16, n_layer=1, n_head=2)
        ).eval()
        probe = copy.deepcopy(model)
        cfg = TorchBPLAConfig(prefix_bits=3, affine_path="float", linear_chunk_out=4)
        diagnostics = AttentionDiagnostics("exact")
        replaced = replace_attention_matmuls(
            probe,
            cfg,
            SharedBPLATables(cfg),
            mode="exact",
            diagnostics=diagnostics,
        )
        input_ids = torch.randint(0, model.config.vocab_size, (1, 8))
        with torch.no_grad():
            expected = model(input_ids).logits
            actual = probe(input_ids).logits
        self.assertEqual(replaced, 1)
        torch.testing.assert_close(actual, expected, atol=1.0e-6, rtol=1.0e-6)
        self.assertTrue(diagnostics.recorded)
        self.assertAlmostEqual(diagnostics.qk_score_mae, 0.0)
        self.assertAlmostEqual(diagnostics.softmax_probability_mae, 0.0)
        self.assertAlmostEqual(diagnostics.attention_output_mae, 0.0)
        self.assertEqual(diagnostics.masked_probability_max, 0.0)

    def test_selective_attention_modes_run(self):
        source = GPT2LMHeadModel(
            GPT2Config(vocab_size=64, n_positions=16, n_ctx=16, n_embd=16, n_layer=1, n_head=2)
        ).eval()
        input_ids = torch.randint(0, source.config.vocab_size, (1, 8))
        cfg = TorchBPLAConfig(prefix_bits=3, affine_path="dyadic", dyadic_terms=2, linear_chunk_out=4)
        for mode in ("bpla-qk", "bpla-pv", "bpla-full"):
            with self.subTest(mode=mode):
                probe = copy.deepcopy(source)
                diagnostics = AttentionDiagnostics(mode)
                replace_attention_matmuls(
                    probe,
                    cfg,
                    SharedBPLATables(cfg),
                    mode=mode,
                    diagnostics=diagnostics,
                )
                with torch.no_grad():
                    logits = probe(input_ids).logits
                self.assertTrue(torch.isfinite(logits).all().item())
                self.assertTrue(diagnostics.recorded)
                self.assertIsNotNone(diagnostics.qk_score_mae)
                self.assertIsNotNone(diagnostics.softmax_probability_mae)
                self.assertIsNotNone(diagnostics.attention_output_mae)


if __name__ == "__main__":
    unittest.main()
