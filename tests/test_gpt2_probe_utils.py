from __future__ import annotations

from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from argparse import Namespace

import torch

from experiments.torch_bpla_gpt2_probe import convert_model, make_dry_run_model, normalize_dataset_name


class GPT2ProbeUtilityTests(unittest.TestCase):
    def test_legacy_wikitext_alias_is_namespaced(self):
        self.assertEqual(normalize_dataset_name("wikitext"), "Salesforce/wikitext")

    def test_explicit_dataset_id_is_unchanged(self):
        self.assertEqual(normalize_dataset_name("owner/dataset"), "owner/dataset")

    def test_activation_only_conversion_reports_replaced_gelus(self):
        model = make_dry_run_model(torch.device("cpu"))
        args = Namespace(
            prefix_bits=4,
            affine_path="dyadic",
            dyadic_terms=2,
            max_shift=16,
            activation_range=4.0,
            linear_chunk_out=4,
            no_conv1d=True,
            no_gelu=False,
            max_conv1d_modules=None,
        )
        _, replaced_conv, replaced_activations = convert_model(model, args)
        self.assertEqual(replaced_conv, 0)
        self.assertEqual(replaced_activations, model.config.n_layer)


if __name__ == "__main__":
    unittest.main()
