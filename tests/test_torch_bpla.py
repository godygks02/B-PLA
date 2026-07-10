from __future__ import annotations

from pathlib import Path
import sys
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from modules.torch_bpla import TorchBPLAActivation, TorchBPLAConfig, TorchBPLALinear, bpla_multiply_torch


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


if __name__ == "__main__":
    unittest.main()
