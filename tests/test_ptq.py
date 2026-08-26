from __future__ import annotations

from pathlib import Path
import sys
import unittest

import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from modules import torch_ptq
from modules.torch_ptq import (
    ActivationObserver,
    TorchPTQConfig,
    TorchPTQLinear,
    calibrate_ptq_model,
    fake_quantize_activation,
    finalize_ptq_model,
    quantize_weight_per_channel,
    replace_ptq_attention_matmuls,
    replace_ptq_gpt2_conv1d,
    replace_ptq_linear,
)


def _relative_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    actual = actual.detach()
    expected = expected.detach()
    return float((actual - expected).norm() / expected.norm())


class WeightQuantizationTests(unittest.TestCase):
    def test_round_trip_stays_within_one_percent(self):
        torch.manual_seed(0)
        weight = torch.randn(128, 256)
        dequantized, _ = quantize_weight_per_channel(weight, bits=8, per_channel=True)
        self.assertLess(_relative_error(dequantized, weight), 0.01)

    def test_error_is_bounded_by_half_a_step(self):
        """Symmetric quantization rounds, so no element may move by more than s/2."""

        torch.manual_seed(1)
        weight = torch.randn(32, 64) * torch.tensor([[3.0]] * 32).cumsum(0)
        dequantized, scale = quantize_weight_per_channel(weight, bits=8, per_channel=True)
        self.assertTrue(bool(((dequantized - weight).abs() <= scale / 2 + 1e-6).all()))

    def test_per_channel_beats_per_tensor(self):
        """The setting exists to make the baseline strong; check it actually does."""

        torch.manual_seed(2)
        # Channel magnitudes spanning three orders of magnitude are what makes a
        # single tensor-wide scale lossy, and transformer projections do this.
        weight = torch.randn(64, 128) * torch.logspace(-2, 1, 64).unsqueeze(1)
        per_channel, _ = quantize_weight_per_channel(weight, bits=8, per_channel=True)
        per_tensor, _ = quantize_weight_per_channel(weight, bits=8, per_channel=False)
        self.assertLess(_relative_error(per_channel, weight), _relative_error(per_tensor, weight))

    def test_zero_channel_does_not_produce_nan(self):
        weight = torch.randn(8, 16)
        weight[3] = 0.0
        dequantized, _ = quantize_weight_per_channel(weight, bits=8)
        self.assertFalse(bool(torch.isnan(dequantized).any()))
        torch.testing.assert_close(dequantized[3], torch.zeros(16))

    def test_more_bits_is_more_accurate(self):
        torch.manual_seed(3)
        weight = torch.randn(16, 64)
        errors = [
            _relative_error(quantize_weight_per_channel(weight, bits=b)[0], weight)
            for b in (4, 6, 8, 10)
        ]
        self.assertEqual(errors, sorted(errors, reverse=True))


class ActivationQuantizationTests(unittest.TestCase):
    def test_signed_grid_has_255_levels_at_8_bits(self):
        x = torch.linspace(-4.0, 4.0, 100000)
        quantized = fake_quantize_activation(x, 4.0, bits=8, signed=True)
        self.assertEqual(int(torch.unique(quantized).numel()), 255)

    def test_unsigned_grid_is_finer_for_non_negative_input(self):
        """Attention probabilities are non-negative; the sign bit is free width."""

        torch.manual_seed(4)
        probabilities = torch.rand(20000)
        amax = float(probabilities.max())
        signed = fake_quantize_activation(probabilities, amax, bits=8, signed=True)
        unsigned = fake_quantize_activation(probabilities, amax, bits=8, signed=False)
        self.assertLess(
            _relative_error(unsigned, probabilities), _relative_error(signed, probabilities)
        )

    def test_values_beyond_the_range_are_clipped_not_wrapped(self):
        x = torch.tensor([-100.0, 0.0, 100.0])
        quantized = fake_quantize_activation(x, 1.0, bits=8, signed=True)
        self.assertLessEqual(float(quantized.abs().max()), 1.0 + 1e-6)

    def test_all_zero_input_is_not_nan(self):
        quantized = fake_quantize_activation(torch.zeros(10), 0.0, bits=8)
        self.assertFalse(bool(torch.isnan(quantized).any()))


class ObserverTests(unittest.TestCase):
    def test_percentile_clips_an_outlier_that_min_max_would_follow(self):
        torch.manual_seed(5)
        values = torch.randn(200000)
        values[0] = 5000.0  # One activation outlier, the transformer failure mode.
        percentile = ActivationObserver(percentile=99.99, bins=2048)
        min_max = ActivationObserver(percentile=100.0, bins=2048)
        percentile.observe(values)
        min_max.observe(values)
        self.assertLess(percentile.resolve(), 100.0)
        self.assertGreater(min_max.resolve(), 4000.0)

    def test_percentile_range_beats_min_max_range_on_outlier_data(self):
        torch.manual_seed(6)
        values = torch.randn(100000)
        values[0] = 5000.0
        percentile = ActivationObserver(percentile=99.9, bins=2048)
        min_max = ActivationObserver(percentile=100.0, bins=2048)
        percentile.observe(values)
        min_max.observe(values)
        by_percentile = fake_quantize_activation(values, percentile.resolve(), 8)
        by_min_max = fake_quantize_activation(values, min_max.resolve(), 8)
        # Compare on the bulk of the distribution: clipping trades a large error
        # on one element for a small error on all the others, which is the point.
        bulk = slice(1, None)
        self.assertLess(
            _relative_error(by_percentile[bulk], values[bulk]),
            _relative_error(by_min_max[bulk], values[bulk]),
        )

    def test_histogram_growth_loses_no_counts(self):
        """Rebinning merges whole bins, so every observation must survive it."""

        torch.manual_seed(7)
        small = torch.randn(50000).abs()
        large = torch.randn(50000).abs() * 40.0
        observer = ActivationObserver(percentile=99.0, bins=2048)
        observer.observe(small)
        observer.observe(large)
        self.assertEqual(float(observer._histogram.sum()), float(observer.count))

    def test_batch_order_shifts_the_range_by_less_than_one_bin(self):
        """Growing from a small first batch overshoots; it may cost only the grid."""

        torch.manual_seed(7)
        small = torch.randn(50000).abs()
        large = torch.randn(50000).abs() * 40.0

        forward = ActivationObserver(percentile=99.0, bins=2048)
        forward.observe(small)
        forward.observe(large)
        backward = ActivationObserver(percentile=99.0, bins=2048)
        backward.observe(large)
        backward.observe(small)
        bin_width = max(forward.upper, backward.upper) / 2048
        self.assertLess(abs(forward.resolve() - backward.resolve()), bin_width)

    def test_resolve_before_observation_raises(self):
        with self.assertRaises(RuntimeError):
            ActivationObserver().resolve()


class PTQLinearTests(unittest.TestCase):
    def _calibrated_linear(self, source: nn.Linear, config: TorchPTQConfig, x: torch.Tensor):
        layer = TorchPTQLinear(source, config)
        torch_ptq.set_ptq_state(layer, torch_ptq.OBSERVING)
        with torch.no_grad():
            layer(x)
        finalize_ptq_model(layer)
        return layer

    def test_tracks_the_exact_layer(self):
        torch.manual_seed(8)
        source = nn.Linear(256, 128)
        x = torch.randn(64, 256)
        layer = self._calibrated_linear(source, TorchPTQConfig(), x)
        with torch.no_grad():
            approximate = layer(x)
        self.assertLess(_relative_error(approximate, source(x)), 1e-2)

    def test_per_channel_weights_beat_per_tensor_in_the_layer(self):
        torch.manual_seed(9)
        source = nn.Linear(128, 64)
        with torch.no_grad():
            source.weight.mul_(torch.logspace(-2, 1, 64).unsqueeze(1))
        x = torch.randn(32, 128)
        reference = source(x)
        per_channel = self._calibrated_linear(source, TorchPTQConfig(per_channel_weights=True), x)
        per_tensor = self._calibrated_linear(source, TorchPTQConfig(per_channel_weights=False), x)
        with torch.no_grad():
            self.assertLess(
                _relative_error(per_channel(x), reference),
                _relative_error(per_tensor(x), reference),
            )

    def test_uncalibrated_forward_raises_instead_of_returning_exact(self):
        """The dangerous failure is a pass-through that looks like a perfect baseline."""

        source = nn.Linear(16, 8)
        layer = TorchPTQLinear(source, TorchPTQConfig())
        with self.assertRaises(RuntimeError):
            layer(torch.randn(4, 16))

    def test_finalize_before_calibration_raises(self):
        layer = TorchPTQLinear(nn.Linear(16, 8), TorchPTQConfig())
        with self.assertRaises(RuntimeError):
            finalize_ptq_model(layer)

    def test_observing_state_is_exact(self):
        """Calibration must see the float model, not a partially quantized one."""

        torch.manual_seed(10)
        source = nn.Linear(32, 16)
        layer = TorchPTQLinear(source, TorchPTQConfig())
        torch_ptq.set_ptq_state(layer, torch_ptq.OBSERVING)
        x = torch.randn(8, 32)
        with torch.no_grad():
            torch.testing.assert_close(layer(x), source(x))

    def test_quantized_output_is_actually_quantized(self):
        """A too-loose tolerance would let a pass-through slip through the suite."""

        torch.manual_seed(11)
        source = nn.Linear(64, 32)
        x = torch.randn(16, 64)
        layer = self._calibrated_linear(source, TorchPTQConfig(), x)
        with torch.no_grad():
            self.assertGreater(_relative_error(layer(x), source(x)), 1e-5)

    def test_token_granularity_helps_when_rows_differ_in_scale(self):
        torch.manual_seed(12)
        source = nn.Linear(128, 64)
        x = torch.randn(32, 128) * torch.logspace(-2, 1, 32).unsqueeze(1)
        reference = source(x)
        per_tensor = self._calibrated_linear(source, TorchPTQConfig(), x)
        per_token = self._calibrated_linear(
            source, TorchPTQConfig(activation_granularity="token"), x
        )
        with torch.no_grad():
            self.assertLess(
                _relative_error(per_token(x), reference),
                _relative_error(per_tensor(x), reference),
            )

    def test_rejects_unknown_granularity(self):
        with self.assertRaises(ValueError):
            TorchPTQLinear(nn.Linear(4, 4), TorchPTQConfig(activation_granularity="channel"))


def _tiny_gpt2():
    from transformers import GPT2Config, GPT2LMHeadModel

    config = GPT2Config(
        vocab_size=64, n_positions=32, n_embd=32, n_layer=2, n_head=2, n_inner=64
    )
    torch.manual_seed(13)
    model = GPT2LMHeadModel(config)
    model.eval()
    return model


class ModelIntegrationTests(unittest.TestCase):
    """End-to-end on a randomly initialized two-layer GPT-2, so no download."""

    def setUp(self):
        self.model = _tiny_gpt2()
        self.inputs = [{"input_ids": torch.randint(0, 64, (2, 16))} for _ in range(2)]
        with torch.no_grad():
            self.reference = self.model(**self.inputs[0]).logits.clone()

    def test_replacement_calibration_and_finalization(self):
        config = TorchPTQConfig()
        converted = replace_ptq_gpt2_conv1d(self.model, config)
        blocks = replace_ptq_attention_matmuls(self.model, config)
        self.assertEqual(converted, 8)  # 4 Conv1D per block, 2 blocks
        self.assertEqual(blocks, 2)

        batches = calibrate_ptq_model(
            self.model, self.inputs, lambda module, batch: module(**batch), 2
        )
        self.assertEqual(batches, 2)
        summary = finalize_ptq_model(self.model)
        self.assertEqual(summary["quantized_layers"], 8)
        self.assertEqual(summary["quantized_attention_blocks"], 2)
        self.assertGreater(summary["activation_range_max"], 0.0)

        with torch.no_grad():
            logits = self.model(**self.inputs[0]).logits
        self.assertFalse(bool(torch.isnan(logits).any()))
        # Close enough to be a working baseline, far enough to be quantized.
        error = _relative_error(logits, self.reference)
        self.assertLess(error, 0.2)
        self.assertGreater(error, 1e-5)

    def test_forward_before_calibration_raises(self):
        config = TorchPTQConfig()
        replace_ptq_gpt2_conv1d(self.model, config)
        replace_ptq_attention_matmuls(self.model, config)
        with self.assertRaises(RuntimeError):
            self.model(**self.inputs[0])

    def test_calibration_pass_reproduces_the_float_model(self):
        config = TorchPTQConfig()
        replace_ptq_gpt2_conv1d(self.model, config)
        replace_ptq_attention_matmuls(self.model, config)
        torch_ptq.set_ptq_state(self.model, torch_ptq.OBSERVING)
        with torch.no_grad():
            logits = self.model(**self.inputs[0]).logits
        # Attention is re-registered through a custom interface, so this also
        # checks the replacement path itself is faithful when not quantizing.
        torch.testing.assert_close(logits, self.reference, rtol=1e-5, atol=1e-5)

    def test_linear_replacer_finds_a_plain_module_tree(self):
        model = nn.Sequential(nn.Linear(8, 8), nn.GELU(), nn.Sequential(nn.Linear(8, 4)))
        replaced = replace_ptq_linear(model, TorchPTQConfig())
        self.assertEqual(replaced, 2)
        # GELU is left exact: W8A8 quantizes weights and activations only.
        self.assertIsInstance(model[1], nn.GELU)


if __name__ == "__main__":
    unittest.main()
