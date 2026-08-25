from __future__ import annotations

from pathlib import Path
import sys
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from modules import torch_pao
from modules.torch_bpla import TorchBPLAConfig, bpla_multiply_torch


def _mogami_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Independent PAM oracle: add the float32 bit patterns as int32.

    This is the trick from Mogami (2020) that the official CUDA kernel uses,
    so agreement with it is evidence that our algebraic implementation of
    Eq. (5)-(8) matches the published operation rather than our reading of it.
    """

    a32 = a.to(torch.float32)
    b32 = b.to(torch.float32)
    bias = torch.tensor(127 << 23, dtype=torch.int32)
    sign_mask = torch.tensor(-(1 << 31), dtype=torch.int32)  # 0x80000000

    ia = a32.view(torch.int32)
    ib = b32.view(torch.int32)
    magnitude = (ia & ~sign_mask) + (ib & ~sign_mask) - bias
    sign = (ia ^ ib) & sign_mask
    out = (magnitude | sign).view(torch.float32)
    zero = (a32 == 0) | (b32 == 0)
    return torch.where(zero, torch.zeros_like(out), out)


class PAOMultiplicationTests(unittest.TestCase):
    def test_matches_mogami_int_addition(self):
        torch.manual_seed(0)
        values = torch.empty(20000).uniform_(-8.0, 8.0).to(torch.float32)
        other = torch.empty(20000).uniform_(-8.0, 8.0).to(torch.float32)
        expected = _mogami_multiply(values, other)
        actual = torch_pao.pao_multiply_torch(values, other)
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=0.0)

    def test_exact_when_either_operand_is_a_power_of_two(self):
        powers = torch.tensor([0.25, 0.5, 1.0, 2.0, 4.0, -8.0])
        others = torch.tensor([1.3, -2.7, 0.4, 3.9, -1.1, 6.25])
        approx = torch_pao.pao_multiply_torch(powers, others)
        torch.testing.assert_close(approx, powers * others, rtol=1e-6, atol=0.0)

    def test_worst_case_relative_error_is_minus_one_ninth(self):
        worst = torch.tensor([1.5])
        approx = torch_pao.pao_multiply_torch(worst, worst)
        relative = (approx - worst * worst) / (worst * worst)
        torch.testing.assert_close(relative, torch.tensor([-1.0 / 9.0]), rtol=1e-6, atol=1e-7)

    def test_relative_error_is_bounded_by_one_ninth(self):
        torch.manual_seed(1)
        a = torch.empty(50000).uniform_(-20.0, 20.0)
        b = torch.empty(50000).uniform_(-20.0, 20.0)
        exact = a * b
        approx = torch_pao.pao_multiply_torch(a, b)
        relative = ((approx - exact) / exact).abs()
        self.assertLessEqual(float(relative.max()), 1.0 / 9.0 + 1e-6)

    def test_zero_operands_produce_zero(self):
        a = torch.tensor([0.0, 3.0, 0.0, -0.0])
        b = torch.tensor([5.0, 0.0, 0.0, 7.0])
        approx = torch_pao.pao_multiply_torch(a, b)
        torch.testing.assert_close(approx, torch.zeros_like(approx))

    def test_pam_equals_zero_plane_bpla_below_mantissa_overflow(self):
        """The two share a first-order form but renormalise differently.

        Dropping the interaction plane from B-PLA leaves ``2^(E1+E2)(1+m1+m2)``,
        which is exactly PAM while ``m1 + m2 < 1``. Above that boundary PAM
        renormalises logarithmically (mantissa ``m1+m2``, exponent ``+1``)
        whereas B-PLA renormalises algebraically (mantissa ``(1+m1+m2)/2``),
        so the degenerate B-PLA form is the weaker of the two there. B-PLA's
        advantage therefore comes from the fitted plane, not from this term.
        """

        torch.manual_seed(2)
        a = torch.empty(8192).uniform_(-6.0, 6.0)
        b = torch.empty(8192).uniform_(-6.0, 6.0)
        config = TorchBPLAConfig(prefix_bits=4, affine_path="float")

        class _ZeroPlaneTables:
            """Both multiplier forms degenerate to ``cross = 0`` when the tile
            centres (separable) or the plane coefficients (legacy) are zero."""

            def multiplier(self, device, dtype):
                segments = 1 << config.prefix_bits
                zeros = torch.zeros(segments, segments, device=device, dtype=dtype)
                return {
                    "centers": torch.zeros(segments, device=device, dtype=dtype),
                    "coeff_a": zeros,
                    "coeff_b": zeros,
                    "coeff_c": zeros,
                }

        degenerate = bpla_multiply_torch(a, b, config, _ZeroPlaneTables())
        pam = torch_pao.pao_multiply_torch(a, b)

        frac_a, _ = torch.frexp(a.abs())
        frac_b, _ = torch.frexp(b.abs())
        below = (frac_a * 2.0 - 1.0) + (frac_b * 2.0 - 1.0) < 1.0
        self.assertTrue(bool(below.any()))
        torch.testing.assert_close(degenerate[below], pam[below])
        self.assertFalse(bool(torch.allclose(degenerate[~below], pam[~below])))

    def test_bpla_multiplier_is_more_accurate_than_pam(self):
        torch.manual_seed(3)
        a = torch.empty(20000).uniform_(-6.0, 6.0)
        b = torch.empty(20000).uniform_(-6.0, 6.0)
        exact = a * b
        pam_error = (torch_pao.pao_multiply_torch(a, b) - exact).abs().mean()
        config = TorchBPLAConfig(prefix_bits=4, affine_path="float")
        bpla_error = (bpla_multiply_torch(a, b, config) - exact).abs().mean()
        self.assertLess(float(bpla_error), float(pam_error))


class PAODerivedOperationTests(unittest.TestCase):
    def test_division_inverts_multiplication_structure(self):
        torch.manual_seed(4)
        a = torch.empty(10000).uniform_(0.1, 12.0)
        b = torch.empty(10000).uniform_(0.1, 12.0)
        quotient = torch_pao.pao_divide_torch(a, b)
        relative = ((quotient - a / b) / (a / b)).abs()
        self.assertLessEqual(float(relative.max()), 1.0 / 8.0 + 1e-5)

    def test_division_by_power_of_two_is_exact(self):
        a = torch.tensor([1.3, -2.7, 5.5, 0.4])
        for divisor in (0.5, 1.0, 2.0, 4.0):
            quotient = torch_pao.pao_divide_torch(a, torch.full_like(a, divisor))
            torch.testing.assert_close(quotient, a / divisor, rtol=1e-6, atol=0.0)

    def test_palog2_and_paexp2_match_definitions(self):
        x = torch.tensor([0.75, 1.0, 1.5, 3.0, 10.0])
        fraction, exponent = torch.frexp(x)
        expected_log = (exponent - 1).to(x.dtype) + (fraction * 2.0 - 1.0)
        torch.testing.assert_close(torch_pao.palog2_torch(x), expected_log)

        y = torch.tensor([-2.5, -0.5, 0.0, 0.5, 2.5, 3.75])
        expected_exp = torch.ldexp(1.0 + (y - torch.floor(y)), torch.floor(y).to(torch.int32))
        torch.testing.assert_close(torch_pao.paexp2_torch(y), expected_exp)

    def test_paexp2_is_exact_at_integers(self):
        y = torch.tensor([-3.0, -1.0, 0.0, 1.0, 5.0])
        torch.testing.assert_close(torch_pao.paexp2_torch(y), torch.exp2(y))

    def test_pasqrt_is_exact_at_even_powers_of_two(self):
        x = torch.tensor([0.25, 1.0, 4.0, 16.0, 64.0])
        torch.testing.assert_close(torch_pao.pasqrt_torch(x), torch.sqrt(x), rtol=1e-6, atol=0.0)

    def test_softmax_rows_are_approximately_normalised(self):
        torch.manual_seed(5)
        logits = torch.randn(64, 32)
        probabilities = torch_pao.pao_softmax_torch(logits, dim=-1)
        self.assertTrue(bool((probabilities >= 0).all()))
        row_sums = probabilities.sum(dim=-1)
        self.assertLess(float((row_sums - 1.0).abs().max()), 0.25)


class PAOModuleTests(unittest.TestCase):
    def test_linear_wrapper_matches_functional_path(self):
        torch.manual_seed(6)
        source = torch.nn.Linear(16, 8)
        x = torch.randn(4, 16)
        config = torch_pao.TorchPAOConfig(matmul_chunk_out=3)
        wrapper = torch_pao.TorchPAOLinear(source, config)
        expected = torch_pao.pao_linear_torch(x, source.weight, source.bias, config)
        torch.testing.assert_close(wrapper(x), expected)

    def test_matmul_matches_chunk_independent_result(self):
        torch.manual_seed(7)
        a = torch.randn(2, 5, 7)
        b = torch.randn(2, 7, 9)
        wide = torch_pao.pao_matmul_torch(a, b, torch_pao.TorchPAOConfig(matmul_chunk_out=64))
        narrow = torch_pao.pao_matmul_torch(a, b, torch_pao.TorchPAOConfig(matmul_chunk_out=2))
        torch.testing.assert_close(wide, narrow)

    def test_layer_norm_wrapper_tracks_exact_layer_norm_loosely(self):
        torch.manual_seed(8)
        source = torch.nn.LayerNorm(32)
        x = torch.randn(4, 32)
        approx = torch_pao.TorchPAOLayerNorm(source, torch_pao.TorchPAOConfig())(x)
        exact = source(x)
        self.assertLess(float((approx - exact).detach().abs().mean()), 0.3)


if __name__ == "__main__":
    unittest.main()
