from __future__ import annotations

from pathlib import Path
import sys
import unittest

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from modules import torch_ptq
from modules.torch_ptq import TorchPTQConfig, finalize_ptq_model
from modules.torch_fqvit import (
    PowerOfTwoFactorObserver,
    TorchFQViTLayerNorm,
    fake_quantize_power_of_two_factor,
    log2_quantize_attention,
    replace_fqvit_layer_norms,
)


class LogIntSoftmaxTests(unittest.TestCase):
    def test_values_land_on_powers_of_two(self):
        torch.manual_seed(0)
        probabilities = torch.rand(5000).clamp(min=1e-6)
        quantized = log2_quantize_attention(probabilities, bits=4)
        nonzero = quantized[quantized > 0]
        exponents = torch.log2(nonzero)
        torch.testing.assert_close(exponents, exponents.round(), rtol=0, atol=1e-6)

    def test_small_probabilities_flush_to_zero(self):
        # 4 bits gives exponents 0..15, so anything under 2^-16 is unrepresentable.
        probabilities = torch.tensor([1.0, 2.0**-15, 2.0**-16, 2.0**-30])
        quantized = log2_quantize_attention(probabilities, bits=4)
        self.assertEqual(float(quantized[0]), 1.0)
        self.assertGreater(float(quantized[1]), 0.0)
        self.assertEqual(float(quantized[3]), 0.0)

    def test_needs_no_calibration(self):
        """Softmax output is bounded to (0,1); that is why LIS is calibration-free."""

        a = log2_quantize_attention(torch.rand(100), bits=4)
        b = log2_quantize_attention(torch.rand(100), bits=4)
        self.assertTrue(bool((a > 0).any()) and bool((b > 0).any()))

    def test_more_bits_reaches_smaller_probabilities(self):
        p = torch.tensor([2.0**-20])
        self.assertEqual(float(log2_quantize_attention(p, bits=4)), 0.0)
        self.assertGreater(float(log2_quantize_attention(p, bits=5)), 0.0)


class PowerOfTwoFactorTests(unittest.TestCase):
    def test_scales_are_power_of_two_divisions_of_one_base(self):
        """The whole point of PTF: one layer scale, per-channel shifts."""

        torch.manual_seed(1)
        x = torch.randn(256, 32) * torch.logspace(-2, 1, 32)
        observer = PowerOfTwoFactorObserver(bits=8)
        observer.observe(x)
        scale, _ = observer.resolve()
        ratios = scale.max() / scale
        torch.testing.assert_close(torch.log2(ratios), torch.log2(ratios).round(), rtol=0, atol=1e-5)
        self.assertLessEqual(float(ratios.max()), 2.0 ** (observer.factors - 1) + 1e-6)

    def test_beats_a_single_layer_wise_scale(self):
        """PTF exists because channel ranges differ; check it actually helps."""

        torch.manual_seed(2)
        x = torch.randn(512, 64) * torch.logspace(-2, 1, 64)
        observer = PowerOfTwoFactorObserver(bits=8)
        observer.observe(x)
        scale, zero_point = observer.resolve()
        ptf = fake_quantize_power_of_two_factor(x, scale, zero_point, 8)
        flat = fake_quantize_power_of_two_factor(
            x, torch.full_like(scale, float(scale.max())), zero_point, 8
        )
        self.assertLess(
            float((ptf - x).pow(2).sum()), float((flat - x).pow(2).sum())
        )

    def test_resolve_before_observation_raises(self):
        with self.assertRaises(RuntimeError):
            PowerOfTwoFactorObserver().resolve()


class FQViTLayerNormTests(unittest.TestCase):
    def test_tracks_the_exact_layer_norm(self):
        torch.manual_seed(3)
        source = nn.LayerNorm(64)
        layer = TorchFQViTLayerNorm(source, TorchPTQConfig())
        x = torch.randn(32, 64)
        torch_ptq.set_ptq_state(layer, torch_ptq.OBSERVING)
        with torch.no_grad():
            observed = layer(x)
            torch.testing.assert_close(observed, source(x))  # observing is exact
        finalize_ptq_model(layer)
        with torch.no_grad():
            quantized = layer(x)
        reference = source(x).detach()
        error = float((quantized - reference).norm() / reference.norm())
        self.assertLess(error, 5e-2)
        self.assertGreater(error, 1e-6)  # and it really did quantize

    def test_uncalibrated_forward_raises(self):
        layer = TorchFQViTLayerNorm(nn.LayerNorm(8), TorchPTQConfig())
        with self.assertRaises(RuntimeError):
            layer(torch.randn(2, 8))

    def test_replacer_walks_the_tree(self):
        model = nn.Sequential(nn.LayerNorm(8), nn.Sequential(nn.LayerNorm(8), nn.GELU()))
        self.assertEqual(replace_fqvit_layer_norms(model, TorchPTQConfig()), 2)
        # GELU is untouched: FQ-ViT never addresses it.
        self.assertIsInstance(model[1][1], nn.GELU)


if __name__ == "__main__":
    unittest.main()
