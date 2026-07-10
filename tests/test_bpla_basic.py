from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from modules import bpla_activation
from modules import bpla_multiplier


class BPLABasicTests(unittest.TestCase):
    def test_bpla_multiplier_runs_and_is_reasonable(self):
        rng = np.random.default_rng(0)
        a = rng.normal(0.0, 1.0, size=2048).astype(np.float32)
        b = rng.normal(0.0, 1.0, size=2048).astype(np.float32)

        ref = bpla_multiplier.exact_multiply(a, b)
        out = bpla_multiplier.bpla_multiply(a, b, prefix_bits=4)
        metrics = bpla_multiplier.error_summary(out, ref)

        self.assertEqual(out.shape, ref.shape)
        self.assertTrue(np.all(np.isfinite(out)))
        self.assertLess(metrics["mean_rel"], 1e-3)

    def test_bpla_multiplier_dyadic_path_runs(self):
        rng = np.random.default_rng(1)
        a = rng.normal(0.0, 1.0, size=2048).astype(np.float32)
        b = rng.normal(0.0, 1.0, size=2048).astype(np.float32)

        ref = bpla_multiplier.exact_multiply(a, b)
        out = bpla_multiplier.bpla_multiply(a, b, prefix_bits=4, affine_path="dyadic", dyadic_terms=2)
        metrics = bpla_multiplier.error_summary(out, ref)

        self.assertEqual(out.shape, ref.shape)
        self.assertTrue(np.all(np.isfinite(out)))
        self.assertLess(metrics["mean_rel"], 1e-2)

    def test_bpla_activation_runs_and_is_reasonable(self):
        x = np.linspace(-4.0, 4.0, 2048)

        ref = bpla_activation.exact_activation(x, "gelu")
        out = bpla_activation.bpla_activation(x, "gelu", prefix_bits=4)
        metrics = bpla_activation.error_summary(out, ref)

        self.assertEqual(out.shape, ref.shape)
        self.assertTrue(np.all(np.isfinite(out)))
        self.assertLess(metrics["mae"], 2e-3)

    def test_bpla_activation_dyadic_path_runs(self):
        x = np.linspace(-4.0, 4.0, 2048)

        ref = bpla_activation.exact_activation(x, "gelu")
        out = bpla_activation.bpla_activation(x, "gelu", prefix_bits=4, affine_path="dyadic", dyadic_terms=3)
        metrics = bpla_activation.error_summary(out, ref)

        self.assertEqual(out.shape, ref.shape)
        self.assertTrue(np.all(np.isfinite(out)))
        self.assertLess(metrics["mae"], 1e-2)


if __name__ == "__main__":
    unittest.main()
