from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from modules.torch_shiftaddllm import (
    binary_code_check,
    conv1d_to_linear,
    cpu_fallback,
    decoder_blocks,
    find_layers,
    import_upstream,
    load_checkpoint_into,
    quantize_model,
    quantized_weights,
    save_checkpoint,
    upstream_args,
    upstream_root,
)


def _tiny_gpt2(seed: int = 0):
    from transformers import GPT2Config, GPT2LMHeadModel

    torch.manual_seed(seed)
    config = GPT2Config(n_layer=2, n_embd=128, n_head=2, n_positions=32, vocab_size=64)
    return GPT2LMHeadModel(config).eval()


def _upstream_available() -> bool:
    return (
        (upstream_root() / "quant_methods" / "shiftaddllm.py").is_file()
        and importlib.util.find_spec("primefac") is not None
        and importlib.util.find_spec("scipy") is not None
    )


class UpstreamArgsTests(unittest.TestCase):
    """The per-mode switches are what separate the two published rows."""

    def test_acc_turns_on_incoherence_processing(self):
        args = upstream_args("acc", 3)
        self.assertTrue(args.pre_rescale and args.pre_proj and args.act_order)
        self.assertFalse(args.block_quant)

    def test_lat_is_blockwise_without_rotation(self):
        args = upstream_args("lat", 3)
        self.assertTrue(args.block_quant and args.act_order and args.use_bst)
        self.assertFalse(args.pre_rescale or args.pre_proj)

    def test_unknown_mode_is_refused(self):
        with self.assertRaises(ValueError):
            upstream_args("fast", 3)


class ConversionTests(unittest.TestCase):
    def test_conv1d_to_linear_is_exact(self):
        model = _tiny_gpt2()
        ids = torch.randint(0, 64, (2, 16))
        with torch.no_grad():
            before = model(ids).logits
            names = conv1d_to_linear(model)
            after = model(ids).logits
        self.assertEqual(len(names), 4 * 2)
        torch.testing.assert_close(after, before, rtol=0, atol=1e-5)
        for block in decoder_blocks(model):
            self.assertTrue(all(isinstance(m, torch.nn.Linear) for m in find_layers(block).values()))

    def test_find_layers_sees_the_four_gpt2_matrices(self):
        block = decoder_blocks(_tiny_gpt2())[0]
        self.assertEqual(sorted(find_layers(block)), ["attn.c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj"])


class BinaryCodeCheckTests(unittest.TestCase):
    def test_a_binary_code_passes(self):
        torch.manual_seed(0)
        rows, columns, wbits = 256, 16, 3
        alpha = torch.rand(8, 1, columns, wbits)  # one scale set per row group and column
        signs = torch.randint(0, 2, (8, rows // 8, columns, wbits)).float() * 2 - 1
        weight = (signs * alpha).sum(-1).reshape(rows, columns)
        result = binary_code_check(weight, wbits)
        self.assertTrue(result["conclusive"])
        self.assertTrue(result["binary_coded"])
        self.assertLessEqual(result["max_distinct_per_group"], 8)

    def test_a_dense_matrix_fails(self):
        torch.manual_seed(0)
        result = binary_code_check(torch.randn(256, 16), 3)
        self.assertTrue(result["conclusive"])
        self.assertFalse(result["binary_coded"])
        self.assertEqual(result["groups_over_limit"], result["groups"])


class CheckpointTests(unittest.TestCase):
    """What the harness loads must land on the right parameters in the right layout."""

    def _saved(self, meta_extra=None):
        model = _tiny_gpt2()
        names = conv1d_to_linear(model)
        for block in decoder_blocks(model):
            for module in find_layers(block).values():
                module.weight.data.mul_(0.5)
        weights = quantized_weights(model, names)
        directory = tempfile.mkdtemp()
        path = Path(directory) / "ckpt.pt"
        meta = {"model_id": "tiny", "partial": False, **(meta_extra or {})}
        save_checkpoint(path, weights, meta)
        return path, weights

    def test_round_trip_restores_conv1d_layout(self):
        path, weights = self._saved()
        fresh = _tiny_gpt2()
        original = fresh.transformer.h[0].attn.c_attn.weight.detach().clone()
        meta, count = load_checkpoint_into(fresh, path, expected_model_id="tiny")
        self.assertEqual(count, 8)
        self.assertEqual(meta["model_id"], "tiny")
        torch.testing.assert_close(fresh.transformer.h[0].attn.c_attn.weight, original * 0.5)
        # Biases, embeddings and the vocabulary projection are not touched.
        self.assertTrue(all(name.endswith(".weight") and ".h." in name for name in weights))

    def test_another_model_is_refused(self):
        path, _ = self._saved()
        with self.assertRaises(ValueError):
            load_checkpoint_into(_tiny_gpt2(), path, expected_model_id="gpt2")

    def test_partial_checkpoint_needs_explicit_permission(self):
        path, _ = self._saved({"partial": True, "blocks_quantized": 1})
        with self.assertRaises(ValueError):
            load_checkpoint_into(_tiny_gpt2(), path)
        load_checkpoint_into(_tiny_gpt2(), path, allow_partial=True)


@unittest.skipUnless(_upstream_available(), "needs ./fetch_shiftaddllm.sh, primefac and scipy")
class UpstreamQuantizationTests(unittest.TestCase):
    """The port drives the authors' quantizer end to end on a tiny GPT-2."""

    @classmethod
    def setUpClass(cls):
        cls.shiftaddllm, cls.bcquantizer = import_upstream()

    def _quantize(self, mode: str):
        torch.manual_seed(0)
        model = _tiny_gpt2()
        conv1d_to_linear(model)
        calibration = [torch.randint(0, 64, (1, 32)) for _ in range(4)]
        args = upstream_args(mode, 3, bcq_round=2)
        with cpu_fallback():
            report = quantize_model(
                model, calibration, args, self.shiftaddllm, self.bcquantizer,
                model_name="tiny", batch_size=2, log=lambda _: None,
            )
        return model, report

    def test_lat_weights_are_binary_codes(self):
        model, report = self._quantize("lat")
        self.assertEqual(len(report), 2)
        for block in decoder_blocks(model):
            for module in find_layers(block).values():
                result = binary_code_check(module.weight.data, 3)
                self.assertTrue(result["conclusive"])
                self.assertTrue(result["binary_coded"])
        with torch.no_grad():
            self.assertTrue(torch.isfinite(model(torch.randint(0, 64, (1, 16))).logits).all())

    def test_acc_weights_are_rotated_back_to_dense(self):
        """--acc binarizes in a rotated basis and undoes the rotation afterwards."""

        model, _ = self._quantize("acc")
        results = [
            binary_code_check(module.weight.data, 3)
            for block in decoder_blocks(model)
            for module in find_layers(block).values()
        ]
        self.assertTrue(all(r["conclusive"] for r in results))
        self.assertTrue(all(not r["binary_coded"] for r in results))
        with torch.no_grad():
            self.assertTrue(torch.isfinite(model(torch.randint(0, 64, (1, 16))).logits).all())


if __name__ == "__main__":
    unittest.main()
