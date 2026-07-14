from __future__ import annotations

from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from modules.compute_energy import (
    BPLAComputeConfig,
    ComputeEnergyTablePJ,
    bpla_gelu_energy_pj,
    bpla_multiplier_energy_pj,
    estimate_workload_compute_energy,
    fp32_gelu_energy_pj,
    mlp_workload,
)


class ComputeEnergyTests(unittest.TestCase):
    def setUp(self):
        self.table = ComputeEnergyTablePJ()
        self.dyadic = BPLAComputeConfig("dyadic", 2, 24)

    def test_two_term_multiplier_counts_current_a_b_c_representation(self):
        result = bpla_multiplier_energy_pj(self.dyadic, self.table)
        self.assertEqual(result["fixed_shift_count"], 6.0)
        self.assertEqual(result["fixed_add_count"], 8.0)
        self.assertAlmostEqual(result["total_pj"], 0.655)
        self.assertLess(result["ratio_to_fp32_mul"], 1.0)

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


if __name__ == "__main__":
    unittest.main()
