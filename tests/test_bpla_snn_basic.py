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


if __name__ == "__main__":
    unittest.main()

