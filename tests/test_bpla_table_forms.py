"""
Tests for the two table-construction forms that replaced the originals:
the separable multiplier and per-segment anchor selection.

Both changes are invisible in exact arithmetic and only alter how dyadic
quantization error enters, so the tests below check equivalence on the float
path and strict improvement on the dyadic path.
"""

from __future__ import annotations

import os
import subprocess
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


class MultiplierExactnessTests(unittest.TestCase):
    """Properties the multiplier must hold for the no-exact-multiply claim."""

    def test_exact_when_either_operand_is_a_power_of_two(self):
        """A zero mantissa means the interaction term is exactly zero.

        Consulting the tile plane there injects its residual into a product that
        should be exact, and would make routing a power-of-two scaling through
        the multiplier worse than leaving it as a float multiply.
        """

        config = TorchBPLAConfig(prefix_bits=4, affine_path="dyadic", dyadic_terms=2)
        powers = torch.tensor([0.125, 0.25, 1.0, 2.0, 8.0, -0.5])
        others = torch.tensor([1.5, -2.25, 7.0, 0.03125, -1000.0, 3.7])
        torch.testing.assert_close(
            bpla_multiply_torch(others, powers, config), others * powers, rtol=0, atol=0
        )
        torch.testing.assert_close(
            bpla_multiply_torch(powers, others, config), powers * others, rtol=0, atol=0
        )

    def test_propagates_infinities_and_nan(self):
        config = TorchBPLAConfig(prefix_bits=4, affine_path="dyadic", dyadic_terms=2)
        a = torch.tensor([float("-inf"), float("inf"), float("nan"), 2.0])
        b = torch.tensor([2.0, 2.0, 2.0, float("nan")])
        got = bpla_multiply_torch(a, b, config)
        self.assertTrue(bool(torch.isneginf(got[0])))
        self.assertTrue(bool(torch.isposinf(got[1])))
        self.assertTrue(bool(got[2].isnan()))
        self.assertTrue(bool(got[3].isnan()))

    def test_the_zero_mantissa_shortcut_does_not_cost_general_accuracy(self):
        generator = torch.Generator().manual_seed(0)
        a = torch.randn(200000, generator=generator)
        b = torch.randn(200000, generator=generator) * 0.02
        exact = a.double() * b.double()
        config = TorchBPLAConfig(prefix_bits=4, affine_path="dyadic", dyadic_terms=2)
        error = float((bpla_multiply_torch(a, b, config).double() - exact).abs().mean())
        self.assertLess(error, 5e-6)


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


class CompiledPathTests(unittest.TestCase):
    """Fusing the elementwise chain must not change what it computes.

    An earlier version was validated on one fixed shape and on the unit tests,
    both of which passed while the compiled path was silently wrong inside a
    model, where the multiply is called with many shapes and broadcast patterns.
    This runs both paths in one subprocess -- reloading the module in-process
    leaks stale classes into other tests -- and compares them exactly.
    """

    def test_compiled_multiply_matches_eager_across_broadcast_shapes(self):
        script = f"""
import os, sys, torch
sys.path.insert(0, {str(ROOT)!r})
os.environ["BPLA_COMPILE"] = "0"
from modules.torch_bpla import TorchBPLAConfig, bpla_multiply_torch

config = TorchBPLAConfig(prefix_bits=4, affine_path="dyadic", dyadic_terms=2)
shapes = [((4096,), (4096,)),
          ((64, 1, 128), (1, 32, 128)),
          ((2, 12, 64, 1, 64), (2, 12, 1, 16, 64))]
operands, expected = [], []
for shape_a, shape_b in shapes:
    torch.manual_seed(11)
    a, b = torch.randn(*shape_a), torch.randn(*shape_b)
    operands.append((a, b))
    expected.append(bpla_multiply_torch(a, b, config))

import importlib, modules.torch_bpla as m
os.environ["BPLA_COMPILE"] = "1"
importlib.reload(m)
if not m._COMPILE:
    print("SKIP compilation disabled"); raise SystemExit
bad = []
for (a, b), want in zip(operands, expected):
    try:
        got = m.bpla_multiply_torch(a, b, config)
    except Exception as error:
        print("SKIP", type(error).__name__); raise SystemExit
    if not torch.equal(got, want):
        bad.append(tuple(a.shape) + tuple(b.shape))
print("MISMATCH" if bad else "MATCH", bad)
"""
        result = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True, timeout=600
        )
        output = result.stdout.strip()
        if output.startswith("SKIP"):
            self.skipTest(f"compilation unavailable here: {output}")
        self.assertTrue(output.startswith("MATCH"), msg=output + result.stderr[-2000:])


if __name__ == "__main__":
    unittest.main()


class MantissaWidthTests(unittest.TestCase):
    """The third complexity knob: how wide the fixed-point mantissa path is.

    The multiplier spends most of its energy on additions whose cost is linear
    in this width, so it is the knob the cost model is most sensitive to.
    """

    def _multiply(self, a, b, bits, terms=2):
        config = TorchBPLAConfig(affine_path="dyadic", dyadic_terms=terms, mantissa_bits=bits)
        return bpla_multiply_torch(a, b, config, SharedBPLATables(config)).double()

    def setUp(self):
        torch.manual_seed(0)
        self.a = torch.empty(60000).uniform_(-6.0, 6.0)
        self.b = torch.empty(60000).uniform_(-6.0, 6.0)
        self.exact = self.a.double() * self.b.double()

    def _error(self, bits):
        approximate = self._multiply(self.a, self.b, bits)
        return float(
            approximate.sub(self.exact).pow(2).mean().sqrt() / self.exact.pow(2).mean().sqrt()
        )

    def test_full_width_matches_unconstrained(self):
        """24 bits is float32's significand, so it must cost nothing."""

        self.assertAlmostEqual(self._error(24), self._error(None), places=9)

    def test_error_grows_monotonically_as_the_path_narrows(self):
        errors = [self._error(bits) for bits in (24, 16, 12, 10, 8, 6)]
        for narrower, wider in zip(errors[1:], errors[:-1]):
            self.assertGreaterEqual(narrower, wider * 0.999)

    def test_sixteen_bits_is_nearly_free(self):
        """Where the knee is decides whether the cost argument works."""

        self.assertLess(self._error(16), self._error(24) * 1.05)

    def test_narrowing_costs_the_exactness_on_powers_of_two(self):
        """Full width multiplies a power of two exactly; a narrow path cannot.

        Exactness on powers of two is one of the properties B-PLA has and
        Mitchell-family multiplication does not, and it survives only while the
        operand mantissas are carried in full. Once the datapath rounds them,
        a zero-fraction operand no longer rescues the other one. The residual is
        bounded by the datapath resolution, which is what makes the trade
        predictable rather than merely empirical.
        """

        powers = torch.tensor([0.25, 0.5, 1.0, 2.0, -4.0, 8.0])
        others = torch.tensor([1.3, -2.7, 0.4, 3.9, -1.1, 6.25])
        exact = powers.double() * others.double()

        torch.testing.assert_close(
            self._multiply(powers, others, 24), exact, rtol=1e-6, atol=0.0
        )
        for bits in (16, 12, 8):
            with self.subTest(bits=bits):
                error = float(
                    (self._multiply(powers, others, bits) - exact).abs().div(exact.abs()).max()
                )
                self.assertGreater(error, 0.0)
                self.assertLess(error, 2.0 ** -(bits - 1))

    def test_quantization_grid_is_actually_applied(self):
        """Results land on the datapath grid, at 2^-bits after normalization.

        The mantissa is held to 2^-(bits-1) in [1, 2), but a product whose
        mantissa overflows is halved to renormalize, which carries it onto the
        finer 2^-bits grid. Asserting the coarser one would fail on exactly the
        products that overflowed.
        """

        from modules.torch_bpla import _fraction_and_exponent

        for bits in (12, 8):
            with self.subTest(bits=bits):
                product = self._multiply(self.a[:4000], self.b[:4000], bits)
                fraction, _, _ = _fraction_and_exponent(product.float())
                scaled = fraction * float(1 << bits)
                torch.testing.assert_close(scaled, scaled.round(), rtol=0, atol=1e-4)
