from __future__ import annotations

from pathlib import Path
import sys
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments import pao_vs_bpla_model as harness


def _unchunked(current: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    """The whole-tensor form the chunked implementation replaced.

    Kept here as an oracle: the chunked version exists only to bound peak
    memory, so any numerical difference between the two is a bug, not a
    trade-off.
    """

    current = current.double()
    reference = reference.double()
    centered_current = current - current.mean(dim=-1, keepdim=True)
    centered_reference = reference - reference.mean(dim=-1, keepdim=True)
    centered_difference = centered_current - centered_reference
    return {
        "logit_mae": float(centered_difference.abs().mean()),
        "logit_rmse": float(centered_difference.pow(2).mean().sqrt()),
        "logit_nrmse": float(
            centered_difference.pow(2).mean().sqrt() / centered_reference.pow(2).mean().sqrt()
        ),
        "argmax_agreement": float(
            (current.argmax(dim=-1) == reference.argmax(dim=-1)).float().mean() * 100.0
        ),
        "output_gain": float(
            (centered_current * centered_reference).sum() / centered_reference.pow(2).sum()
        ),
        "uncentered_logit_mae": float((current - reference).abs().mean()),
        "uncentered_output_gain": float((current * reference).sum() / reference.pow(2).sum()),
    }


class ChunkedComparisonTests(unittest.TestCase):
    def setUp(self):
        self._chunk = harness._COMPARISON_CHUNK_ELEMENTS
        self.addCleanup(setattr, harness, "_COMPARISON_CHUNK_ELEMENTS", self._chunk)

    def _assert_matches(self, current, reference, chunk_elements):
        harness._COMPARISON_CHUNK_ELEMENTS = chunk_elements
        actual = harness.compare_to_reference(current, reference)
        expected = _unchunked(current, reference)
        self.assertEqual(set(actual), set(expected))
        for key, value in expected.items():
            self.assertAlmostEqual(
                actual[key], value, delta=max(abs(value), 1.0) * 1e-12, msg=key
            )

    def test_matches_the_unchunked_form_across_chunk_counts(self):
        torch.manual_seed(0)
        reference = torch.randn(97, 733) * 5.0
        current = reference + torch.randn(97, 733) * 0.01
        for chunk_elements in (733, 733 * 7, 733 * 96, 733 * 97, 733 * 500):
            with self.subTest(chunk_elements=chunk_elements):
                self._assert_matches(current, reference, chunk_elements)

    def test_a_single_row_chunk_is_still_correct(self):
        """The chunk floor is one row; a vocabulary wider than the budget hits it."""

        torch.manual_seed(1)
        reference = torch.randn(11, 4096)
        current = reference + torch.randn(11, 4096) * 0.05
        self._assert_matches(current, reference, 1)

    def test_identical_inputs_give_perfect_scores(self):
        torch.manual_seed(2)
        logits = torch.randn(64, 512)
        harness._COMPARISON_CHUNK_ELEMENTS = 512 * 5
        result = harness.compare_to_reference(logits, logits)
        self.assertAlmostEqual(result["logit_nrmse"], 0.0, places=12)
        self.assertAlmostEqual(result["argmax_agreement"], 100.0, places=12)
        self.assertAlmostEqual(result["output_gain"], 1.0, places=12)

    def test_agreement_counts_rows_not_elements(self):
        """A row-level metric divided by an element count would read ~0%."""

        reference = torch.tensor([[3.0, 1.0, 0.0], [0.0, 2.0, 1.0]])
        current = torch.tensor([[3.0, 1.0, 0.0], [0.0, 0.0, 5.0]])
        harness._COMPARISON_CHUNK_ELEMENTS = 3
        self.assertAlmostEqual(
            harness.compare_to_reference(current, reference)["argmax_agreement"], 50.0
        )

    def test_centering_ignores_a_per_row_constant(self):
        """Softmax cannot see a per-row shift, so the centered metrics must not."""

        torch.manual_seed(3)
        # float64 inputs, so the assertion tests the centering rather than the
        # rounding: adding a shift of ~100 to a float32 logit and subtracting it
        # again already costs 1e-6 of relative accuracy before centering runs.
        reference = torch.randn(32, 256, dtype=torch.float64)
        shifted = reference + torch.randn(32, 1, dtype=torch.float64) * 100.0
        harness._COMPARISON_CHUNK_ELEMENTS = 256 * 4
        result = harness.compare_to_reference(shifted, reference)
        self.assertAlmostEqual(result["logit_nrmse"], 0.0, places=10)
        self.assertAlmostEqual(result["output_gain"], 1.0, places=10)
        # The uncentered figure does see it, which is why both are reported.
        self.assertGreater(result["uncentered_logit_mae"], 1.0)


if __name__ == "__main__":
    unittest.main()
