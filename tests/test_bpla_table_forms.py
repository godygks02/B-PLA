"""
Tests for the two table-construction forms that replaced the originals:
the separable multiplier and per-segment anchor selection.

Both changes are invisible in exact arithmetic and only alter how dyadic
quantization error enters, so the tests below check equivalence on the float
path and strict improvement on the dyadic path.
"""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from modules.torch_bpla import (
    SharedBPLATables,
    TorchBPLAConfig,
    TorchBPLAConv2d,
    TorchBPLALinear,
    _functional_bpla,
    bpla_multiply_torch,
)


def _operands(count: int = 40000, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    a = torch.randn(count, generator=generator)
    b = torch.randn(count, generator=generator) * 0.02
    keep = (a != 0) & (b != 0)
    return a[keep], b[keep]


class SeparableMultiplierTests(unittest.TestCase):
    def test_agrees_with_the_plane_form_in_exact_arithmetic(self):
        """nu*m1 + mu*(m2-nu) is an identity for nu*m1 + mu*m2 - mu*nu."""

        a, b = _operands()
        for prefix_bits in (2, 4, 6):
            separable = TorchBPLAConfig(
                prefix_bits=prefix_bits, affine_path="float", multiplier_form="separable"
            )
            plane = TorchBPLAConfig(
                prefix_bits=prefix_bits, affine_path="float", multiplier_form="plane"
            )
            torch.testing.assert_close(
                bpla_multiply_torch(a, b, separable),
                bpla_multiply_torch(a, b, plane),
                rtol=1e-5,
                atol=1e-9,
            )

    def test_never_worse_than_the_plane_form_under_quantization(self):
        a, b = _operands()
        exact = a.double() * b.double()
        for prefix_bits in (2, 3, 4):
            for terms in (1, 2, 3, 4):
                errors = {}
                for form in ("plane", "separable"):
                    config = TorchBPLAConfig(
                        prefix_bits=prefix_bits,
                        affine_path="dyadic",
                        dyadic_terms=terms,
                        multiplier_form=form,
                    )
                    approx = bpla_multiply_torch(a, b, config).double()
                    errors[form] = float((approx - exact).abs().mean())
                # Once the term budget is large enough for both forms to reach
                # the exact plane, the remaining difference is float32
                # association order, so compare with a tolerance well below any
                # meaningful regression.
                self.assertLessEqual(
                    errors["separable"],
                    errors["plane"] * 1.001,
                    msg=f"separable regressed at k={prefix_bits}, T={terms}: {errors}",
                )

    def test_table_is_one_array_of_tile_centres(self):
        """The separable form must not materialize the 2^k x 2^k planes."""

        config = TorchBPLAConfig(prefix_bits=4, affine_path="dyadic", multiplier_form="separable")
        table = SharedBPLATables(config).multiplier(torch.device("cpu"), torch.float32)
        self.assertIn("centers", table)
        self.assertNotIn("coeff_c", table)
        self.assertEqual(table["centers"].numel(), 1 << config.prefix_bits)

    def test_beats_piecewise_affine_multiplication_at_every_term_budget(self):
        """The regression this fix was made for: at T=1 the old form lost to PAM."""

        from modules.torch_pao import TorchPAOConfig, pao_multiply_torch

        a, b = _operands()
        exact = a.double() * b.double()
        relative = lambda x: ((x.double() - exact) / exact.abs().clamp_min(1e-30)).abs()
        pam = float(relative(pao_multiply_torch(a, b, TorchPAOConfig())).max())

        for terms in (1, 2, 3, 4):
            config = TorchBPLAConfig(
                prefix_bits=4,
                affine_path="dyadic",
                dyadic_terms=terms,
                multiplier_form="separable",
            )
            worst = float(relative(bpla_multiply_torch(a, b, config)).max())
            self.assertLess(worst, pam, msg=f"T={terms}: {worst:.4f} vs PAM {pam:.4f}")


class AnchorSelectionTests(unittest.TestCase):
    DOMAINS = {
        "exp2_fraction": (lambda g: torch.rand(200000, generator=g), torch.exp2),
        "reciprocal_unit_mantissa": (
            lambda g: 1.0 + torch.rand(200000, generator=g),
            torch.reciprocal,
        ),
        "rsqrt_mantissa": (lambda g: 0.5 + 1.5 * torch.rand(200000, generator=g), torch.rsqrt),
    }

    def _error(self, name: str, config: TorchBPLAConfig) -> float:
        generator = torch.Generator().manual_seed(3)
        sampler, exact_fn = self.DOMAINS[name]
        x = sampler(generator)
        exact = exact_fn(x)
        approx = _functional_bpla(x, name, config, SharedBPLATables(config))
        return float((approx - exact).pow(2).mean().sqrt() / exact.pow(2).mean().sqrt())

    def test_auto_is_never_worse_than_the_legacy_intercept_form(self):
        for name in self.DOMAINS:
            for terms in (1, 2, 3, 4):
                common = dict(prefix_bits=4, affine_path="dyadic", dyadic_terms=terms)
                auto = self._error(name, TorchBPLAConfig(**common, anchor_mode="auto"))
                legacy = self._error(name, TorchBPLAConfig(**common, anchor_mode="intercept"))
                self.assertLessEqual(
                    auto, legacy * 1.001, msg=f"{name} T={terms}: auto {auto} vs legacy {legacy}"
                )

    def test_auto_keeps_the_intercept_where_the_domain_abuts_the_origin(self):
        """exp2 lives on [0,1), so the intercept is the well-conditioned choice."""

        config = TorchBPLAConfig(prefix_bits=4, affine_path="dyadic", dyadic_terms=2)
        table = SharedBPLATables(config).functional(
            "exp2_fraction", torch.device("cpu"), torch.float32
        )
        self.assertEqual(table["anchor_mode"], "intercept")

    def test_auto_moves_the_anchor_where_the_domain_is_far_from_the_origin(self):
        """1/u lives on [1,2), where the intercept is a long extrapolation."""

        config = TorchBPLAConfig(prefix_bits=4, affine_path="dyadic", dyadic_terms=2)
        table = SharedBPLATables(config).functional(
            "reciprocal_unit_mantissa", torch.device("cpu"), torch.float32
        )
        self.assertNotEqual(table["anchor_mode"], "intercept")

    def test_reciprocal_and_rsqrt_now_beat_the_pao_equivalent_at_two_terms(self):
        """The regression this fix was made for: both used to need T>=3."""

        from modules.torch_pao import TorchPAOConfig, pao_divide_torch, pasqrt_torch

        pao_config = TorchPAOConfig()
        generator = torch.Generator().manual_seed(3)
        cases = {
            "reciprocal_unit_mantissa": (
                1.0 + torch.rand(200000, generator=generator),
                torch.reciprocal,
                lambda x: pao_divide_torch(torch.ones_like(x), x, pao_config),
            ),
            "rsqrt_mantissa": (
                0.5 + 1.5 * torch.rand(200000, generator=generator),
                torch.rsqrt,
                lambda x: pao_divide_torch(
                    torch.ones_like(x), pasqrt_torch(x, pao_config), pao_config
                ),
            ),
        }
        config = TorchBPLAConfig(prefix_bits=4, affine_path="dyadic", dyadic_terms=2)
        for name, (x, exact_fn, pao_fn) in cases.items():
            exact = exact_fn(x)
            scale = exact.pow(2).mean().sqrt()
            bpla = float((_functional_bpla(x, name, config, SharedBPLATables(config)) - exact).pow(2).mean().sqrt() / scale)
            pao = float((pao_fn(x) - exact).pow(2).mean().sqrt() / scale)
            self.assertLess(bpla, pao, msg=f"{name}: B-PLA {bpla:.4e} vs PAO {pao:.4e}")


class CoverageGapTests(unittest.TestCase):
    """The two module types the replacers used to walk past."""

    def test_conv2d_proxy_tracks_the_exact_convolution(self):
        torch.manual_seed(0)
        config = TorchBPLAConfig(prefix_bits=8, affine_path="float")
        for kernel, stride, padding in ((16, 16, 0), (3, 1, 1), (5, 2, 2)):
            source = torch.nn.Conv2d(3, 8, kernel_size=kernel, stride=stride, padding=padding)
            x = torch.randn(2, 3, 32, 32)
            approx = TorchBPLAConv2d(source, config, SharedBPLATables(config))(x)
            exact = source(x)
            self.assertEqual(approx.shape, exact.shape)
            relative = float((approx - exact).detach().abs().max() / exact.detach().abs().max())
            self.assertLess(relative, 1e-4, msg=f"k={kernel} s={stride} p={padding}: {relative}")

    def test_conv2d_proxy_rejects_grouped_convolutions(self):
        config = TorchBPLAConfig(prefix_bits=4, affine_path="float")
        grouped = torch.nn.Conv2d(4, 8, kernel_size=3, groups=2)
        with self.assertRaises(NotImplementedError):
            TorchBPLAConv2d(grouped, config)

    def test_linear_replacement_can_share_a_tied_weight(self):
        """GPT-2 ties lm_head to the embedding; cloning would break it and cost 154 MB."""

        source = torch.nn.Linear(8, 16, bias=False)
        config = TorchBPLAConfig(prefix_bits=4, affine_path="float")

        shared = TorchBPLALinear(source, config, share_weight=True)
        self.assertEqual(shared.weight.data_ptr(), source.weight.data_ptr())

        copied = TorchBPLALinear(source, config, share_weight=False)
        self.assertNotEqual(copied.weight.data_ptr(), source.weight.data_ptr())
        torch.testing.assert_close(copied.weight, source.weight)

    def test_replacers_leave_the_extra_module_types_alone_by_default(self):
        """Enabling the wider scope must stay an explicit, reported choice."""

        from modules.torch_bpla import replace_linear_and_gelu

        config = TorchBPLAConfig(prefix_bits=4, affine_path="float")
        module = torch.nn.Sequential(torch.nn.Conv2d(3, 4, 3), torch.nn.Linear(4, 4))

        replace_linear_and_gelu(module, config, replace_gelu=False)
        self.assertIsInstance(module[0], torch.nn.Conv2d)
        self.assertIsInstance(module[1], TorchBPLALinear)

        module = torch.nn.Sequential(torch.nn.Conv2d(3, 4, 3), torch.nn.Linear(4, 4))
        replace_linear_and_gelu(module, config, replace_gelu=False, replace_conv2d=True)
        self.assertIsInstance(module[0], TorchBPLAConv2d)


if __name__ == "__main__":
    unittest.main()
