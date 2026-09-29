from __future__ import annotations

from pathlib import Path
import sys
import unittest

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from modules.torch_smoothquant import (
    ChannelMaxObserver,
    SmoothQuantConfig,
    _scale_weight_inputs,
    _weight_input_maxima,
    smooth_group,
)

try:
    from transformers.pytorch_utils import Conv1D
except Exception:  # pragma: no cover - transformers is a hard dependency elsewhere
    Conv1D = None


def _linear_block(width: int = 16, out: int = 24):
    torch.manual_seed(0)
    norm = nn.LayerNorm(width)
    nn.init.uniform_(norm.weight, 0.5, 1.5)
    nn.init.uniform_(norm.bias, -0.5, 0.5)
    projection = nn.Linear(width, out)
    return norm, projection


class WeightLayoutTests(unittest.TestCase):
    """Linear stores [out, in] and Conv1D stores [in, out]; the fold must follow.

    Getting this backwards is silent -- both orientations broadcast without an
    error and only the numbers are wrong -- so the layouts are pinned directly.
    """

    def test_linear_input_maxima_are_per_column(self):
        layer = nn.Linear(4, 3)
        layer.weight.data = torch.tensor(
            [[1.0, -9.0, 2.0, 0.0], [3.0, 1.0, -4.0, 0.5], [0.0, 2.0, 1.0, -7.0]]
        )
        torch.testing.assert_close(
            _weight_input_maxima(layer), torch.tensor([3.0, 9.0, 4.0, 7.0])
        )

    @unittest.skipIf(Conv1D is None, "transformers not available")
    def test_conv1d_input_maxima_are_per_row(self):
        layer = Conv1D(3, 4)  # nf=3 outputs, nx=4 inputs -> weight is [4, 3]
        layer.weight.data = torch.tensor(
            [[1.0, 3.0, 0.0], [-9.0, 1.0, 2.0], [2.0, -4.0, 1.0], [0.0, 0.5, -7.0]]
        )
        torch.testing.assert_close(
            _weight_input_maxima(layer), torch.tensor([3.0, 9.0, 4.0, 7.0])
        )

    @unittest.skipIf(Conv1D is None, "transformers not available")
    def test_scaling_matches_across_layouts(self):
        torch.manual_seed(1)
        width, out = 6, 5
        linear = nn.Linear(width, out, bias=False)
        conv = Conv1D(out, width)
        conv.weight.data = linear.weight.data.t().contiguous()
        scale = torch.linspace(0.5, 2.0, width)

        _scale_weight_inputs(linear, scale)
        _scale_weight_inputs(conv, scale)
        torch.testing.assert_close(conv.weight.data, linear.weight.data.t())


class SmoothingIdentityTests(unittest.TestCase):
    def test_smoothing_leaves_the_composed_output_unchanged(self):
        """The whole point: migration is an identity on the exact model.

        If it were not, every SmoothQuant row would be measuring the migration
        as well as the quantization, and the comparison against ptq-w8a8 would
        no longer isolate the method.
        """

        norm, projection = _linear_block()
        x = torch.randn(32, 16) * 3.0
        before = projection(norm(x))

        activation_max = norm(x).abs().amax(dim=0)
        smooth_group(norm, [projection], activation_max, SmoothQuantConfig(alpha=0.5))
        after = projection(norm(x))

        torch.testing.assert_close(after, before, rtol=1e-5, atol=1e-5)

    def test_identity_holds_for_every_alpha(self):
        for alpha in (0.0, 0.25, 0.5, 0.85, 1.0):
            norm, projection = _linear_block()
            x = torch.randn(16, 16) * 2.0
            before = projection(norm(x))
            smooth_group(
                norm, [projection], norm(x).abs().amax(dim=0), SmoothQuantConfig(alpha=alpha)
            )
            torch.testing.assert_close(
                projection(norm(x)), before, rtol=1e-5, atol=1e-5,
                msg=f"identity broken at alpha={alpha}",
            )

    def test_migration_flattens_activations_and_loads_the_weights(self):
        """Direction check: the outlier must move from activations to weights."""

        norm, projection = _linear_block()
        x = torch.randn(64, 16)
        x[:, 3] *= 40.0  # one outlier channel, the situation SmoothQuant targets

        activations_before = norm(x).abs().amax(dim=0)
        weights_before = _weight_input_maxima(projection)
        smooth_group(norm, [projection], activations_before, SmoothQuantConfig(alpha=0.5))
        activations_after = norm(x).abs().amax(dim=0)
        weights_after = _weight_input_maxima(projection)

        spread = lambda v: float(v.max() / v.mean())
        self.assertLess(spread(activations_after), spread(activations_before))
        self.assertGreater(spread(weights_after), spread(weights_before))

    def test_endpoint_alphas_reduce_to_the_closed_form(self):
        """s = a^alpha / w^(1-alpha) collapses at both ends of the range."""

        config = SmoothQuantConfig()
        floor = config.min_scale

        # alpha = 0 -> s = 1 / w
        norm, projection = _linear_block()
        weights_before = _weight_input_maxima(projection)
        x = torch.randn(16, 16)
        scale = smooth_group(
            norm, [projection], norm(x).abs().amax(dim=0), SmoothQuantConfig(alpha=0.0)
        )
        torch.testing.assert_close(
            scale, (1.0 / weights_before.clamp(min=floor)).clamp(min=floor)
        )

        # alpha = 1 -> s = a
        norm, projection = _linear_block()
        x = torch.randn(16, 16)
        activation_max = norm(x).abs().amax(dim=0)
        scale = smooth_group(
            norm, [projection], activation_max, SmoothQuantConfig(alpha=1.0)
        )
        torch.testing.assert_close(scale, activation_max.clamp(min=floor))

    def test_a_group_shares_one_scale(self):
        """q, k and v hang off one LayerNorm and must be folded together."""

        torch.manual_seed(2)
        norm = nn.LayerNorm(8)
        projections = [nn.Linear(8, 8, bias=False) for _ in range(3)]
        x = torch.randn(20, 8) * 2.0
        before = [p(norm(x)) for p in projections]

        smooth_group(norm, projections, norm(x).abs().amax(dim=0), SmoothQuantConfig())

        for projection, reference in zip(projections, before):
            torch.testing.assert_close(projection(norm(x)), reference, rtol=1e-5, atol=1e-5)

    def test_zero_channel_does_not_divide_by_zero(self):
        norm, projection = _linear_block()
        activation_max = norm(torch.randn(8, 16)).abs().amax(dim=0)
        activation_max[0] = 0.0
        projection.weight.data[:, 0] = 0.0
        scale = smooth_group(norm, [projection], activation_max, SmoothQuantConfig())
        self.assertTrue(torch.isfinite(scale).all())
        self.assertTrue(torch.isfinite(norm.weight).all())


class _PreNormBlock(nn.Module):
    """A GPT-2/ViT-shaped block: residual comes from the pre-norm tensor."""

    def __init__(self, width: int = 12, heads: int = 3):
        super().__init__()
        self.norm_attention = nn.LayerNorm(width)
        self.query = nn.Linear(width, width)
        self.key = nn.Linear(width, width)
        self.value = nn.Linear(width, width)
        self.projection = nn.Linear(width, width)
        self.norm_mlp = nn.LayerNorm(width)
        self.up = nn.Linear(width, 4 * width)
        self.down = nn.Linear(4 * width, width)

    def forward(self, x):
        normalized = self.norm_attention(x)
        attention = self.query(normalized) + self.key(normalized) + self.value(normalized)
        x = x + self.projection(attention)
        return x + self.down(torch.relu(self.up(self.norm_mlp(x))))


class _TinyTransformer(nn.Module):
    def __init__(self, depth: int = 3, width: int = 12):
        super().__init__()
        self.layers = nn.ModuleList([_PreNormBlock(width) for _ in range(depth)])
        self.final_norm = nn.LayerNorm(width)          # feeds the head, not a block
        self.head = nn.Linear(width, 5)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.head(self.final_norm(x))


class GroupDiscoveryTests(unittest.TestCase):
    """The wiring is found from the dataflow, not from attribute names.

    The first version of this module walked fixed paths and broke as soon as the
    library renamed a field, so what is pinned here is that discovery survives
    names it has never seen.
    """

    def _discover(self, model):
        from modules.torch_smoothquant import discover_groups

        return discover_groups(model, torch.randn(4, 12), lambda m, b: m(b))

    def test_finds_two_groups_per_block(self):
        model = _TinyTransformer(depth=3)
        groups = self._discover(model)
        self.assertEqual(len(groups), 6)               # 3 blocks x {attention, mlp}

    def test_attention_group_holds_query_key_value_together(self):
        model = _TinyTransformer(depth=1)
        groups = self._discover(model)
        block = model.layers[0]
        by_norm = {id(norm): projections for norm, projections in groups}
        attention = by_norm[id(block.norm_attention)]
        self.assertEqual(len(attention), 3)
        self.assertEqual(
            {id(m) for m in attention},
            {id(block.query), id(block.key), id(block.value)},
        )
        self.assertEqual([id(m) for m in by_norm[id(block.norm_mlp)]], [id(block.up)])

    def test_the_model_final_norm_is_left_alone(self):
        """SmoothQuant folds inside blocks; the head's norm is not a block."""

        model = _TinyTransformer(depth=2)
        groups = self._discover(model)
        self.assertNotIn(id(model.final_norm), {id(norm) for norm, _ in groups})

    def test_discovered_groups_smooth_to_an_identity(self):
        torch.manual_seed(3)
        model = _TinyTransformer(depth=2)
        x = torch.randn(8, 12) * 3.0
        before = model(x)

        from modules.torch_smoothquant import smooth_model

        summary = smooth_model(
            model, [x], lambda m, b: m(b), max_batches=1, config=SmoothQuantConfig()
        )
        torch.testing.assert_close(model(x), before, rtol=1e-4, atol=1e-4)
        self.assertEqual(summary["smoothquant_groups"], 4)
        self.assertEqual(summary["smoothquant_projections"], 8)   # 2 x (3 qkv + 1 up)


class ChannelMaxObserverTests(unittest.TestCase):
    def test_accumulates_the_maximum_over_batches(self):
        observer = ChannelMaxObserver()
        observer.observe(torch.tensor([[1.0, -5.0, 2.0]]))
        observer.observe(torch.tensor([[-9.0, 1.0, 0.5]]))
        torch.testing.assert_close(observer.maximum, torch.tensor([9.0, 5.0, 2.0]))

    def test_folds_leading_dimensions(self):
        observer = ChannelMaxObserver()
        observer.observe(torch.randn(4, 7, 5))
        self.assertEqual(tuple(observer.maximum.shape), (5,))


if __name__ == "__main__":
    unittest.main()
