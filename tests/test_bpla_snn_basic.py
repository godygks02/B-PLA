from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from modules import bpla_activation
from modules import bpla_SNN_activation
from modules import bpla_SNN_multiplier
from modules import bpla_multiplier
from modules import pla_snn


class BPLASNNBasicTests(unittest.TestCase):
    def test_snn_activation_fs_and_if_run(self):
        x = np.linspace(-4.0, 4.0, 512)
        ref = bpla_activation.exact_activation(x, "gelu")

        fs_cfg = bpla_SNN_activation.BPLASpikingNeuronConfig(neuron_type="fs")
        if_cfg = bpla_SNN_activation.BPLASpikingNeuronConfig(neuron_type="if")
        fs_out = bpla_SNN_activation.bpla_snn_activation(x, fs_cfg)["decoded"]
        if_out = bpla_SNN_activation.bpla_snn_activation(x, if_cfg)["decoded"]
        fs_metrics = bpla_SNN_activation.error_summary(fs_out, ref)
        if_metrics = bpla_SNN_activation.error_summary(if_out, ref)

        self.assertEqual(fs_out.shape, ref.shape)
        self.assertEqual(if_out.shape, ref.shape)
        self.assertTrue(np.all(np.isfinite(fs_out)))
        self.assertTrue(np.all(np.isfinite(if_out)))
        self.assertLess(fs_metrics["mae"], 1.0e-2)
        self.assertLessEqual(fs_metrics["mae"], if_metrics["mae"] + 1.0e-3)

    def test_snn_multiplier_fs_runs(self):
        rng = np.random.default_rng(11)
        a = rng.normal(0.0, 1.0, size=512).astype(np.float32)
        b = rng.normal(0.0, 1.0, size=512).astype(np.float32)
        ref = bpla_multiplier.exact_multiply(a, b)

        cfg = bpla_SNN_multiplier.BPLASpikingMultiplierConfig(neuron_type="fs", mantissa_bits=12)
        out = bpla_SNN_multiplier.bpla_snn_multiply(a, b, cfg)["decoded"]
        metrics = bpla_SNN_multiplier.error_summary(out, ref)

        self.assertEqual(out.shape, ref.shape)
        self.assertTrue(np.all(np.isfinite(out)))
        self.assertLess(metrics["mean_rel"], 5.0e-2)

    def test_bitplane_prefix_shape(self):
        x = np.array([0.0, 0.5, -1.25, 2.0])
        enc = bpla_SNN_activation.encode_bitplane_spikes(
            x,
            bpla_SNN_activation.BitPlaneEncodingConfig(bit_width=8, fractional_bits=4),
        )
        prefix = bpla_SNN_activation.prefix_index_from_bitplanes(enc["spikes"], 3)
        self.assertEqual(prefix.shape, x.shape)
        self.assertTrue(np.all(prefix >= 0))

    def test_term_free_event_affine_matches_compiled_pla(self):
        slopes = np.array([0.375, -0.625], dtype=np.float64)
        biases = np.array([0.125, -0.25], dtype=np.float64)
        bit_positions = np.array([2, 1, 0], dtype=np.int64)
        compiled = pla_snn.compile_affine_synapses(
            slopes,
            biases,
            bit_positions,
            input_fractional_bits=3,
        )
        spikes = np.array([[1, 0, 1], [0, 1, 1]], dtype=np.uint8)
        signs = np.array([1, -1], dtype=np.int64)
        indices = np.array([0, 1], dtype=np.int64)
        decoded, event_adds = pla_snn.event_affine_accumulate(
            spikes,
            signs,
            compiled.increments[indices],
            compiled.bias[indices],
        )
        x = np.array([5.0 / 8.0, -3.0 / 8.0])
        expected = slopes[indices] * x + biases[indices]
        np.testing.assert_allclose(decoded, expected, atol=1.0e-6)
        np.testing.assert_array_equal(event_adds, np.array([2, 2]))

    def test_snn_paths_report_zero_runtime_multiplications(self):
        x = np.linspace(-2.0, 2.0, 32)
        act = bpla_SNN_activation.bpla_snn_activation(x)
        self.assertEqual(act["ops"]["runtime_multiplications"], 0.0)

        a = np.linspace(0.25, 1.25, 32, dtype=np.float32)
        b = np.linspace(-1.0, 1.0, 32, dtype=np.float32)
        mul = bpla_SNN_multiplier.bpla_snn_multiply(a, b)
        self.assertEqual(mul["ops"]["runtime_multiplications"], 0.0)


if __name__ == "__main__":
    unittest.main()
