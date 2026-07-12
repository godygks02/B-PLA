from __future__ import annotations

from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.torch_bpla_gpt2_probe import normalize_dataset_name


class GPT2ProbeUtilityTests(unittest.TestCase):
    def test_legacy_wikitext_alias_is_namespaced(self):
        self.assertEqual(normalize_dataset_name("wikitext"), "Salesforce/wikitext")

    def test_explicit_dataset_id_is_unchanged(self):
        self.assertEqual(normalize_dataset_name("owner/dataset"), "owner/dataset")


if __name__ == "__main__":
    unittest.main()
