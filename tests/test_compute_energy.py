from __future__ import annotations

from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from modules.compute_energy import (
    BPLAComputeConfig,
    ComputeEnergyTablePJ,
    bpla_coefficient_bits,
    bpla_gelu_energy_pj,
    bpla_layernorm_energy_pj,
    bpla_multiplier_energy_pj,
    bpla_softmax_energy_pj,
    chen_pam_multiplier_energy_pj,
    estimate_workload_compute_energy,
    fp32_gelu_energy_pj,
    fp32_layernorm_energy_pj,
    fp32_softmax_energy_pj,
    mlp_workload,
    multiplier_cost_summary,
)


class ComputeEnergyTests(unittest.TestCase):
    def setUp(self):
        self.table = ComputeEnergyTablePJ()
        self.dyadic = BPLAComputeConfig("dyadic", 2, 24)
        # The composed Softmax / LayerNorm / workload totals below were pinned
        # under the legacy plane-form multiplier count (8 additions at T = 2);
        # they test the composition, so they keep that form explicitly.
        self.legacy = BPLAComputeConfig("dyadic", 2, 24, "plane")

    def test_two_term_separable_multiplier_counts_two_products_and_a_subtraction(self):
        # nu*m1 + mu*(m2 - nu): 2T table-driven shifts, and 3 + 2T addends plus
        # the subtraction, so 2T + 3 additions. Shifts are priced at zero.
        result = bpla_multiplier_energy_pj(self.dyadic, self.table)
        self.assertEqual(result["multiplier_form"], "separable")
        self.assertEqual(result["fixed_shift_count"], 0.0)
        self.assertEqual(result["variable_shift_count"], 4.0)
        self.assertEqual(result["fixed_add_count"], 7.0)
        self.assertAlmostEqual(result["total_pj"], 0.58)
        self.assertLess(result["ratio_to_fp32_mul"], 1.0)

    def test_legacy_plane_form_still_reproduces_the_old_a_b_c_counts(self):
        result = bpla_multiplier_energy_pj(BPLAComputeConfig("dyadic", 2, 24, "plane"), self.table)
        self.assertEqual(result["fixed_shift_count"], 6.0)
        self.assertEqual(result["fixed_add_count"], 8.0)
        self.assertAlmostEqual(result["total_pj"], 0.655)

    def test_variable_shifts_can_be_charged(self):
        priced = ComputeEnergyTablePJ(variable_shift=0.01)
        free = bpla_multiplier_energy_pj(self.dyadic, self.table)["total_pj"]
        charged = bpla_multiplier_energy_pj(self.dyadic, priced)["total_pj"]
        self.assertAlmostEqual(charged - free, 4 * 0.01)
        # Chen-PAM has no table-driven shift, so the price never reaches it.
        self.assertAlmostEqual(
            chen_pam_multiplier_energy_pj(4, 24, priced)["total_pj"],
            chen_pam_multiplier_energy_pj(4, 24, self.table)["total_pj"],
        )

    def test_chen_pam_level_cost_is_linear_in_the_level(self):
        # Level 0 is the single-tile plane 1.5x + 1.5y - 2.25; every level adds
        # one full-width accumulation and one narrower residual addition.
        level0 = chen_pam_multiplier_energy_pj(0, 24, self.table)
        self.assertEqual(level0["full_add_count"], 4.0)
        self.assertEqual(level0["narrow_add_count"], 0.0)
        self.assertEqual(level0["variable_shift_count"], 0.0)
        previous = level0["total_pj"]
        for level in range(1, 6):
            result = chen_pam_multiplier_energy_pj(level, 24, self.table)
            self.assertEqual(result["full_add_count"], 4.0 + level)
            self.assertEqual(result["narrow_add_widths"], [24.0 - i + 1 for i in range(1, level + 1)])
            self.assertGreater(result["total_pj"], previous)
            previous = result["total_pj"]
        # Level 4 versus the T = 2 separable evaluator at the same width: the
        # exact-coefficient plane costs more additions than two SPT terms.
        self.assertGreater(
            chen_pam_multiplier_energy_pj(4, 24, self.table)["fixed_add_count"],
            bpla_multiplier_energy_pj(self.dyadic, self.table)["fixed_add_count"],
        )
        with self.assertRaises(ValueError):
            chen_pam_multiplier_energy_pj(-1, 24, self.table)

    def test_coefficient_storage_is_the_separable_table(self):
        # 2^k centres, T terms, 1 sign bit + 5 shift-index bits per term.
        self.assertEqual(bpla_coefficient_bits(4, 2), 16 * 2 * 6)
        self.assertEqual(bpla_coefficient_bits(5, 3), 32 * 3 * 6)

    def test_cost_summary_covers_the_weighted_scope_backends(self):
        pam = multiplier_cost_summary("chen-pam", 4, 2, None)
        spt = multiplier_cost_summary("bpla-dyadic", 4, 2, None)
        narrow = multiplier_cost_summary("bpla-dyadic", 4, 2, 12)
        self.assertEqual(pam["coefficient_bits"], 0.0)
        self.assertEqual(spt["coefficient_bits"], 192.0)
        self.assertEqual(spt["mantissa_bits"], 24.0)
        self.assertLess(narrow["energy_pj"], spt["energy_pj"])
        self.assertLess(spt["energy_pj"], pam["energy_pj"])
        self.assertAlmostEqual(spt["energy_over_int8_mul"], spt["energy_pj"] / 0.2)
        with self.assertRaises(ValueError):
            multiplier_cost_summary("pao", 4, 2, None)

    def test_float_affine_multiplier_is_not_mistaken_for_multiplierless(self):
        result = bpla_multiplier_energy_pj(BPLAComputeConfig("float", 2), self.table)
        self.assertEqual(result["fp32_mul_count"], 2.0)
        self.assertGreater(result["total_pj"], self.table.fp32_mul)

    def test_gelu_baseline_is_conservative_tanh_lower_bound(self):
        baseline = fp32_gelu_energy_pj(self.table)
        bpla = bpla_gelu_energy_pj(self.dyadic, self.table)
        self.assertEqual(baseline["fp32_mul_count"], 6.0)
        self.assertEqual(baseline["fp32_add_count"], 2.0)
        self.assertEqual(baseline["tanh_energy_pj"], 0.0)
        self.assertAlmostEqual(baseline["total_pj"], 24.0)
        self.assertEqual(bpla["fixed_shift_count"], 4.0)
        self.assertEqual(bpla["fixed_add_count"], 3.0)

    def test_mlp_counts_and_selective_replacement(self):
        workload = mlp_workload(4, 3, 2, max_linear_modules=1)
        self.assertEqual(workload.multiply_sites, 27)
        self.assertEqual(workload.bpla_multiply_sites, 12)
        self.assertEqual(workload.gelu_sites, 6)
        result = estimate_workload_compute_energy(workload, self.dyadic, self.table)
        self.assertLess(result["bpla_total_pj"], result["ann_total_pj"])

    def test_softmax_energy_matches_composed_correction_path(self):
        baseline = fp32_softmax_energy_pj(elements=4, rows=1, table=self.table)
        bpla = bpla_softmax_energy_pj(4, 1, self.legacy, self.table)
        self.assertAlmostEqual(baseline["total_pj"], 39.615)
        self.assertAlmostEqual(bpla["total_pj"], 30.435)
        self.assertGreater(bpla["correction_normalize_pj"], 0.0)

    def test_layernorm_energy_counts_mean_variance_and_affine(self):
        baseline = fp32_layernorm_energy_pj(elements=4, rows=1, table=self.table)
        bpla = bpla_layernorm_energy_pj(4, 1, self.legacy, self.table)
        self.assertEqual(baseline["fp32_mul_sites"], 14.0)
        self.assertEqual(baseline["fp32_add_sites"], 15.0)
        self.assertEqual(bpla["bpla_multiply_sites"], 14.0)
        self.assertAlmostEqual(baseline["total_pj"], 69.0)
        self.assertAlmostEqual(bpla["total_pj"], 22.9)

    def test_workload_total_includes_normalization_energy(self):
        from modules.compute_energy import ComputeWorkload

        workload = ComputeWorkload(
            multiply_sites=0,
            bpla_multiply_sites=0,
            gelu_sites=0,
            bpla_gelu_sites=0,
            label="normalization-only",
            softmax_rows=1,
            softmax_elements=4,
            bpla_softmax_rows=1,
            bpla_softmax_elements=4,
            layernorm_rows=1,
            layernorm_elements=4,
            bpla_layernorm_rows=1,
            bpla_layernorm_elements=4,
        )
        result = estimate_workload_compute_energy(workload, self.legacy, self.table)
        self.assertAlmostEqual(result["ann_total_pj"], 108.615)
        self.assertAlmostEqual(result["bpla_total_pj"], 53.335)
        self.assertAlmostEqual(result["ann_softmax_pj"], 39.615)
        self.assertAlmostEqual(result["bpla_variant_layernorm_pj"], 22.9)


if __name__ == "__main__":
    unittest.main()
