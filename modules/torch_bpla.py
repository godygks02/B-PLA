"""
CUDA-friendly B-PLA proxy operators for large-model sensitivity tests.

These operators are not a replacement for the hardware-faithful NumPy modules.
They avoid Python/NumPy round-trips so that pretrained PyTorch models can be
probed on CPU or CUDA. The goal is to answer: "Does the model tolerate this
class of B-PLA approximation?"
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.pytorch_utils import Conv1D
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention

try:
    from transformers import AttentionMaskInterface
    from transformers.masking_utils import eager_mask
except ImportError:  # Compatibility with pre-mask-registry Transformers.
    AttentionMaskInterface = None
    eager_mask = None

try:
    # Transformers versions before the ViT attention refactor expose the
    # dispatching module as ViTSelfAttention.
    from transformers.models.vit.modeling_vit import ViTSelfAttention as ViTAttentionModule
except ImportError:
    # Newer versions fold self-attention into ViTAttention.
    from transformers.models.vit.modeling_vit import ViTAttention as ViTAttentionModule


@dataclass(frozen=True)
class TorchBPLAConfig:
    prefix_bits: int = 4
    affine_path: str = "float"
    dyadic_terms: int = 2
    max_shift: int = 16
    activation_range: float = 4.0
    activation_samples_per_segment: int = 64
    linear_chunk_out: int = 32


AttentionMode = Literal["exact", "bpla-qk", "bpla-pv", "bpla-full"]


@dataclass
class AttentionDiagnostics:
    """First-call comparison between a custom attention path and exact matmul."""

    mode: AttentionMode
    recorded: bool = False
    layer_index: int | None = None
    qk_score_mae: float | None = None
    softmax_probability_mae: float | None = None
    attention_output_mae: float | None = None
    masked_probability_max: float | None = None

    def record(
        self,
        attention_module: nn.Module,
        selected_scores: torch.Tensor,
        exact_scores: torch.Tensor,
        selected_probabilities: torch.Tensor,
        exact_probabilities: torch.Tensor,
        selected_output: torch.Tensor,
        exact_output: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> None:
        if self.recorded:
            return
        self.layer_index = getattr(attention_module, "layer_idx", None)
        self.qk_score_mae = float((selected_scores - exact_scores).abs().mean().item())
        self.softmax_probability_mae = float(
            (selected_probabilities - exact_probabilities).abs().mean().item()
        )
        self.attention_output_mae = float((selected_output - exact_output).abs().mean().item())
        self.masked_probability_max = _masked_probability_max(selected_probabilities, attention_mask)
        self.recorded = True


def _masked_probability_max(
    probabilities: torch.Tensor,
    attention_mask: torch.Tensor | None,
) -> float | None:
    """Return the largest probability at additive-mask positions, if present."""

    if attention_mask is None or not torch.is_floating_point(attention_mask):
        return None
    masked = attention_mask < -1.0e4
    if not bool(masked.any().item()):
        return None
    masked = masked.expand_as(probabilities)
    return float(probabilities.masked_select(masked).abs().max().item())


class SharedBPLATables:
    """Model-scoped cache shared by every converted B-PLA operator."""

    def __init__(self, config: TorchBPLAConfig):
        self.config = config
        self._multiplier: dict[tuple[str, torch.dtype], dict[str, torch.Tensor]] = {}
        self._activation: dict[tuple[str, str, torch.dtype], dict[str, torch.Tensor | int | float]] = {}
        self._functional: dict[tuple[str, str, torch.dtype], dict[str, torch.Tensor | float]] = {}

    @staticmethod
    def _device_key(device: torch.device) -> str:
        return str(device)

    def multiplier(self, device: torch.device, dtype: torch.dtype) -> dict[str, torch.Tensor]:
        key = (self._device_key(device), dtype)
        if key not in self._multiplier:
            segments = 1 << self.config.prefix_bits
            centers = (torch.arange(segments, device=device, dtype=dtype) + 0.5) / float(segments)
            mu = centers[:, None]
            nu = centers[None, :]
            self._multiplier[key] = {
                "coeff_a": _maybe_dyadic(nu.expand(segments, segments).contiguous(), self.config),
                "coeff_b": _maybe_dyadic(mu.expand(segments, segments).contiguous(), self.config),
                "coeff_c": _maybe_dyadic(-(mu * nu), self.config),
            }
        return self._multiplier[key]

    def activation(self, target_name: str, device: torch.device, dtype: torch.dtype) -> dict[str, torch.Tensor | int | float]:
        key = (target_name, self._device_key(device), dtype)
        if key not in self._activation:
            self._activation[key] = build_activation_table_torch(target_name, self.config, device, dtype)
        return self._activation[key]

    def functional(self, target_name: str, device: torch.device, dtype: torch.dtype) -> dict[str, torch.Tensor | float]:
        key = (target_name, self._device_key(device), dtype)
        if key not in self._functional:
            self._functional[key] = _build_functional_table(target_name, self.config, device, dtype)
        return self._functional[key]


def _validate_config(config: TorchBPLAConfig) -> None:
    if not 1 <= config.prefix_bits <= 10:
        raise ValueError("prefix_bits must be in [1, 10].")
    if config.affine_path not in {"float", "dyadic"}:
        raise ValueError("affine_path must be 'float' or 'dyadic'.")
    if config.dyadic_terms <= 0:
        raise ValueError("dyadic_terms must be positive.")
    if config.max_shift < 0:
        raise ValueError("max_shift must be non-negative.")


def _signed_pot_quantize(values: torch.Tensor, terms: int, max_shift: int) -> torch.Tensor:
    approx = torch.zeros_like(values)
    min_term = 2.0 ** -max_shift
    for _ in range(terms):
        residual = values - approx
        active = residual.abs() >= 0.5 * min_term
        shift = torch.round(-torch.log2(residual.abs().clamp_min(torch.finfo(values.dtype).tiny)))
        shift = shift.clamp(0, max_shift)
        term = residual.sign() * torch.pow(torch.tensor(2.0, device=values.device, dtype=values.dtype), -shift)
        approx = approx + torch.where(active, term, torch.zeros_like(term))
    return approx


def _maybe_dyadic(values: torch.Tensor, config: TorchBPLAConfig) -> torch.Tensor:
    if config.affine_path == "float":
        return values
    return _signed_pot_quantize(values, config.dyadic_terms, config.max_shift)


def _fraction_and_exponent(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    abs_x = x.abs()
    mant, exponent = torch.frexp(abs_x)
    normal = abs_x > 0
    fraction = torch.where(normal, mant * 2.0 - 1.0, torch.zeros_like(abs_x))
    unbiased_exponent = exponent - 1
    sign = torch.signbit(x)
    return fraction, unbiased_exponent, sign


def bpla_multiply_torch(
    a: torch.Tensor,
    b: torch.Tensor,
    config: TorchBPLAConfig,
    tables: SharedBPLATables | None = None,
) -> torch.Tensor:
    """Approximate elementwise multiplication with torch-native B-PLA logic."""

    _validate_config(config)
    dtype = torch.promote_types(a.dtype, b.dtype)
    a = a.to(dtype)
    b = b.to(dtype)

    frac_a, exp_a, sign_a = _fraction_and_exponent(a)
    frac_b, exp_b, sign_b = _fraction_and_exponent(b)
    segments = 1 << config.prefix_bits
    idx_a = torch.clamp((frac_a * segments).floor().to(torch.long), 0, segments - 1)
    idx_b = torch.clamp((frac_b * segments).floor().to(torch.long), 0, segments - 1)

    shared = tables or SharedBPLATables(config)
    lut = shared.multiplier(a.device, dtype)
    coeff_a = lut["coeff_a"][idx_a, idx_b]
    coeff_b = lut["coeff_b"][idx_a, idx_b]
    coeff_c = lut["coeff_c"][idx_a, idx_b]

    cross = coeff_a * frac_a + coeff_b * frac_b + coeff_c
    mantissa = 1.0 + frac_a + frac_b + cross
    overflow = mantissa >= 2.0
    mantissa = torch.where(overflow, mantissa * 0.5, mantissa)
    exponent = exp_a + exp_b + overflow.to(exp_a.dtype)
    magnitude = torch.ldexp(mantissa, exponent)
    signed = torch.where(sign_a ^ sign_b, -magnitude, magnitude)
    return torch.where((a != 0) & (b != 0), signed, torch.zeros_like(signed))


def bpla_linear_torch(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    config: TorchBPLAConfig,
    tables: SharedBPLATables | None = None,
) -> torch.Tensor:
    """Linear layer using B-PLA elementwise products, chunked over outputs."""

    _validate_config(config)
    original_shape = x.shape[:-1]
    x_flat = x.reshape(-1, x.shape[-1])
    rows = []
    chunk = max(1, config.linear_chunk_out)
    for start in range(0, weight.shape[0], chunk):
        w = weight[start : start + chunk]
        products = bpla_multiply_torch(x_flat[:, None, :], w[None, :, :], config, tables)
        out = products.sum(dim=-1)
        if bias is not None:
            out = out + bias[start : start + chunk]
        rows.append(out)
    return torch.cat(rows, dim=-1).reshape(*original_shape, weight.shape[0])


def bpla_matmul_torch(
    a: torch.Tensor,
    b: torch.Tensor,
    config: TorchBPLAConfig,
    tables: SharedBPLATables | None = None,
) -> torch.Tensor:
    """Batched B-PLA matmul for tensors shaped ``[..., M, K] @ [..., K, N]``."""

    _validate_config(config)
    if a.ndim < 2 or b.ndim < 2:
        raise ValueError("B-PLA matmul inputs must have at least two dimensions.")
    if a.shape[-1] != b.shape[-2]:
        raise ValueError(f"Incompatible B-PLA matmul shapes: {tuple(a.shape)} and {tuple(b.shape)}")

    outputs = []
    chunk = max(1, config.linear_chunk_out)
    for start in range(0, b.shape[-1], chunk):
        b_chunk = b[..., :, start : start + chunk]
        products = bpla_multiply_torch(
            a.unsqueeze(-2),
            b_chunk.transpose(-1, -2).unsqueeze(-3),
            config,
            tables,
        )
        outputs.append(products.sum(dim=-1))
    return torch.cat(outputs, dim=-1)


_FUNCTIONAL_TARGETS: dict[str, tuple[float, float, Callable[[torch.Tensor], torch.Tensor]]] = {
    "exp2_fraction": (0.0, 1.0, torch.exp2),
    # The frexp mantissa is doubled into [1, 2) before this lookup. Keeping
    # the reciprocal table on that interval bounds both slope and intercept
    # so short dyadic coefficient expansions remain useful.
    "reciprocal_unit_mantissa": (1.0, 2.0, torch.reciprocal),
    "rsqrt_mantissa": (0.5, 2.0, torch.rsqrt),
}


def _build_functional_table(
    target_name: str,
    config: TorchBPLAConfig,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, torch.Tensor | float]:
    """Build a prefix-indexed secant table for a bounded scalar function."""

    if target_name not in _FUNCTIONAL_TARGETS:
        raise ValueError(f"Unknown functional target: {target_name}.")
    x_min, x_max, target = _FUNCTIONAL_TARGETS[target_name]
    segments = 1 << config.prefix_bits
    edges = torch.linspace(x_min, x_max, segments + 1, device=device, dtype=dtype)
    y_edges = target(edges)
    slopes = (y_edges[1:] - y_edges[:-1]) / (edges[1:] - edges[:-1])
    intercepts = y_edges[:-1] - slopes * edges[:-1]
    return {
        "slopes": _maybe_dyadic(slopes, config),
        "intercepts": _maybe_dyadic(intercepts, config),
        "x_min": x_min,
        "x_max": x_max,
    }


def _functional_bpla(
    x: torch.Tensor,
    target_name: str,
    config: TorchBPLAConfig,
    tables: SharedBPLATables,
) -> torch.Tensor:
    """Evaluate a bounded B-PLA nonlinear table using fixed-point prefix routing."""

    table = tables.functional(target_name, x.device, x.dtype)
    x_min = float(table["x_min"])
    x_max = float(table["x_max"])
    x_clip = x.clamp(x_min, x_max)
    segments = 1 << config.prefix_bits
    index = ((x_clip - x_min) * (segments / (x_max - x_min))).floor().long()
    index = index.clamp(0, segments - 1)
    slopes = table["slopes"]
    intercepts = table["intercepts"]
    assert isinstance(slopes, torch.Tensor)
    assert isinstance(intercepts, torch.Tensor)
    return slopes[index] * x_clip + intercepts[index]


def bpla_softmax_torch(
    x: torch.Tensor,
    dim: int = -1,
    config: TorchBPLAConfig | None = None,
    tables: SharedBPLATables | None = None,
) -> torch.Tensor:
    """Compose B-PLA exp2, reciprocal, and multiplication into Softmax.

    The maximum subtraction and reductions remain exact control/addition paths.
    Powers of two are reconstructed with ``ldexp``, the software analogue of a
    hardware exponent shift.
    """

    config = config or TorchBPLAConfig()
    _validate_config(config)
    shared = tables or SharedBPLATables(config)
    output_dtype = x.dtype
    work = x.float() if x.dtype in {torch.float16, torch.bfloat16} else x
    finite_row = torch.isfinite(work).any(dim=dim, keepdim=True)
    safe = torch.where(finite_row, work, torch.zeros_like(work))
    shifted = safe - safe.max(dim=dim, keepdim=True).values

    base2 = shifted * 1.4426950408889634
    integer = torch.floor(base2)
    fraction = base2 - integer
    fractional_exp = _functional_bpla(fraction, "exp2_fraction", config, shared)
    # Avoid integer conversion overflow for additive attention masks and make
    # values far below the row maximum exact zeros, as stable Softmax does.
    active = shifted > -80.0
    exponent = integer.clamp(-126.0, 0.0).to(torch.int32)
    exp_values = torch.where(active, torch.ldexp(fractional_exp, exponent), torch.zeros_like(work))

    denominator = exp_values.sum(dim=dim, keepdim=True)
    mantissa, exponent_sum = torch.frexp(denominator)
    reciprocal_unit = _functional_bpla(mantissa * 2.0, "reciprocal_unit_mantissa", config, shared)
    reciprocal = torch.ldexp(reciprocal_unit, 1 - exponent_sum)
    probabilities = bpla_multiply_torch(exp_values, reciprocal, config, shared)
    # One normalization correction reuses the same reciprocal/multiply
    # composition. This materially limits accumulated dyadic table error while
    # keeping the path free of an exact division.
    probability_sum = probabilities.sum(dim=dim, keepdim=True)
    sum_mantissa, sum_exponent = torch.frexp(probability_sum)
    correction_unit = _functional_bpla(
        sum_mantissa * 2.0,
        "reciprocal_unit_mantissa",
        config,
        shared,
    )
    correction = torch.ldexp(correction_unit, 1 - sum_exponent)
    probabilities = bpla_multiply_torch(probabilities, correction, config, shared)
    probabilities = torch.where(finite_row, probabilities, torch.zeros_like(probabilities))
    return probabilities.to(output_dtype)


def bpla_layer_norm_torch(
    x: torch.Tensor,
    normalized_shape: tuple[int, ...],
    weight: torch.Tensor | None,
    bias: torch.Tensor | None,
    eps: float,
    config: TorchBPLAConfig,
    tables: SharedBPLATables,
) -> torch.Tensor:
    """Compose B-PLA multiplication and reciprocal-square-root into LayerNorm."""

    _validate_config(config)
    if tuple(x.shape[-len(normalized_shape) :]) != tuple(normalized_shape):
        raise ValueError(f"Expected trailing shape {normalized_shape}, got {tuple(x.shape)}.")
    output_dtype = x.dtype
    work = x.float() if x.dtype in {torch.float16, torch.bfloat16} else x
    dims = tuple(range(work.ndim - len(normalized_shape), work.ndim))
    element_count = 1
    for size in normalized_shape:
        element_count *= size
    inv_count = torch.tensor(1.0 / element_count, device=work.device, dtype=work.dtype)

    mean = bpla_multiply_torch(work.sum(dim=dims, keepdim=True), inv_count, config, tables)
    centered = work - mean
    squared = bpla_multiply_torch(centered, centered, config, tables)
    variance = bpla_multiply_torch(squared.sum(dim=dims, keepdim=True), inv_count, config, tables)
    variance_eps = variance.clamp_min(0.0) + eps

    mantissa, exponent = torch.frexp(variance_eps)
    odd_exponent = torch.remainder(exponent, 2) != 0
    adjusted_mantissa = torch.where(odd_exponent, mantissa * 2.0, mantissa)
    adjusted_exponent = exponent - odd_exponent.to(exponent.dtype)
    inv_sqrt_mantissa = _functional_bpla(adjusted_mantissa, "rsqrt_mantissa", config, tables)
    inv_std = torch.ldexp(inv_sqrt_mantissa, torch.div(-adjusted_exponent, 2, rounding_mode="floor"))

    normalized = bpla_multiply_torch(centered, inv_std, config, tables)
    if weight is not None:
        normalized = bpla_multiply_torch(normalized, weight.to(normalized.dtype), config, tables)
    if bias is not None:
        normalized = normalized + bias.to(normalized.dtype)
    return normalized.to(output_dtype)


def replace_attention_matmuls(
    module: nn.Module,
    config: TorchBPLAConfig,
    tables: SharedBPLATables,
    mode: AttentionMode = "bpla-full",
    diagnostics: AttentionDiagnostics | None = None,
    approximate_softmax: bool = False,
) -> int:
    """Install an exact or selectively approximated ViT/GPT-2 attention path."""

    valid_modes = {"exact", "bpla-qk", "bpla-pv", "bpla-full"}
    if mode not in valid_modes:
        raise ValueError(f"Unknown attention mode {mode!r}. Choose one of {sorted(valid_modes)}.")
    if diagnostics is not None and diagnostics.mode != mode:
        raise ValueError("diagnostics.mode must match the requested attention mode.")

    attention_modules = [child for child in module.modules() if isinstance(child, (ViTAttentionModule, GPT2Attention))]
    if not attention_modules:
        return 0

    interface_name = f"bpla_{id(tables)}"

    def bpla_attention_forward(
        attention_module: nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None,
        scaling: float | None = None,
        dropout: float = 0.0,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del kwargs
        if scaling is None:
            scaling = query.size(-1) ** -0.5

        use_bpla_qk = mode in {"bpla-qk", "bpla-full"}
        use_bpla_pv = mode in {"bpla-pv", "bpla-full"}
        if use_bpla_qk:
            attention_scores = bpla_matmul_torch(query, key.transpose(-1, -2), config, tables)
        else:
            attention_scores = torch.matmul(query, key.transpose(-1, -2))
        attention_scores = attention_scores * scaling

        attention_weights = attention_scores
        if attention_mask is not None:
            attention_weights = attention_weights + attention_mask
        if approximate_softmax:
            attention_weights = bpla_softmax_torch(attention_weights, dim=-1, config=config, tables=tables)
        else:
            attention_weights = nn.functional.softmax(attention_weights, dim=-1)
        attention_weights = attention_weights.type(value.dtype)
        attention_probabilities = attention_weights
        attention_weights = nn.functional.dropout(
            attention_weights,
            p=dropout,
            training=attention_module.training,
        )
        if use_bpla_pv:
            attention_output = bpla_matmul_torch(attention_weights, value, config, tables)
        else:
            attention_output = torch.matmul(attention_weights, value)

        if diagnostics is not None and not diagnostics.recorded:
            exact_scores = torch.matmul(query, key.transpose(-1, -2)) * scaling
            exact_weights = exact_scores
            if attention_mask is not None:
                exact_weights = exact_weights + attention_mask
            exact_probabilities = nn.functional.softmax(exact_weights, dim=-1).type(value.dtype)
            exact_output = torch.matmul(exact_probabilities, value)
            diagnostics.record(
                attention_module=attention_module,
                selected_scores=attention_scores,
                exact_scores=exact_scores,
                selected_probabilities=attention_probabilities,
                exact_probabilities=exact_probabilities,
                selected_output=attention_output,
                exact_output=exact_output,
                attention_mask=attention_mask,
            )
        return attention_output.transpose(1, 2), attention_weights

    ALL_ATTENTION_FUNCTIONS.register(interface_name, bpla_attention_forward)
    # A custom attention backend needs a mask formatter registered under the
    # same name. Without this, recent Transformers versions deliberately skip
    # causal-mask creation and pass attention_mask=None.
    if AttentionMaskInterface is not None and eager_mask is not None:
        AttentionMaskInterface.register(interface_name, eager_mask)
    for attention_module in attention_modules:
        attention_module.config._attn_implementation = interface_name
    module._bpla_attention_mode = mode
    module._bpla_softmax_enabled = approximate_softmax
    module._bpla_attention_diagnostics = diagnostics
    return len(attention_modules)


def _gelu(x: torch.Tensor) -> torch.Tensor:
    return F.gelu(x, approximate="tanh")


TARGETS: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "gelu": _gelu,
    "quick_gelu": lambda x: x * torch.sigmoid(1.702 * x),
    "relu": F.relu,
    "sigmoid": torch.sigmoid,
    "tanh": torch.tanh,
}


def calibrate_model_activation_range(
    model: nn.Module,
    batches: Iterable[Any],
    forward_batch: Callable[[nn.Module, Any], Any],
    max_batches: int,
) -> float:
    """Measure one symmetric GELU input range across the whole exact model."""

    max_abs = 0.0
    hooks: list[Any] = []
    restored: list[tuple[nn.Module, Any]] = []

    def observe(x: torch.Tensor) -> None:
        nonlocal max_abs
        if x.numel():
            value = x.detach().abs().amax().item()
            max_abs = max(max_abs, float(value))

    def pre_hook(_module: nn.Module, inputs: tuple[Any, ...]) -> None:
        if inputs and isinstance(inputs[0], torch.Tensor):
            observe(inputs[0])

    for child in model.modules():
        child_name = child.__class__.__name__.lower()
        if isinstance(child, nn.GELU) or "geluactivation" in child_name:
            hooks.append(child.register_forward_pre_hook(pre_hook))
        act = getattr(child, "intermediate_act_fn", None)
        if act is not None and callable(act) and not isinstance(act, nn.Module):
            original = act

            def wrapped(x: torch.Tensor, fn: Callable[[torch.Tensor], torch.Tensor] = original) -> torch.Tensor:
                observe(x)
                return fn(x)

            child.intermediate_act_fn = wrapped
            restored.append((child, original))

    try:
        model.eval()
        with torch.no_grad():
            for batch_index, batch in enumerate(batches):
                if batch_index >= max_batches:
                    break
                forward_batch(model, batch)
    finally:
        for hook in hooks:
            hook.remove()
        for child, original in restored:
            child.intermediate_act_fn = original

    if max_abs <= 0.0:
        raise RuntimeError("No GELU inputs were observed during calibration.")
    return max_abs


def build_activation_table_torch(
    target_name: str,
    config: TorchBPLAConfig,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, torch.Tensor | int | float]:
    if target_name not in TARGETS:
        raise ValueError(f"Unknown target_name: {target_name}.")
    _validate_config(config)

    x_min = -float(config.activation_range)
    x_max = float(config.activation_range)
    min_e_routing = -5
    max_e_routing = int(torch.floor(torch.log2(torch.tensor(max(abs(x_min), abs(x_max)), dtype=dtype))).item())
    exponent_bins = max_e_routing - min_e_routing + 1
    segments = 1 + 2 * exponent_bins * (1 << config.prefix_bits)

    xs = torch.linspace(x_min, x_max, max(segments * config.activation_samples_per_segment, 4096), device=device, dtype=dtype)
    idx = activation_prefix_index_torch(xs, config, min_e_routing, max_e_routing)
    ys = TARGETS[target_name](xs)
    slopes = torch.zeros(segments, device=device, dtype=dtype)
    intercepts = torch.zeros(segments, device=device, dtype=dtype)

    for seg in range(segments):
        mask = idx == seg
        x_seg = xs[mask]
        if x_seg.numel() >= 2:
            y_seg = ys[mask]
            x_mean = x_seg.mean()
            y_mean = y_seg.mean()
            denom = ((x_seg - x_mean) ** 2).sum().clamp_min(torch.finfo(dtype).eps)
            slope = ((x_seg - x_mean) * (y_seg - y_mean)).sum() / denom
            intercept = y_mean - slope * x_mean
            slopes[seg] = slope
            intercepts[seg] = intercept
        elif x_seg.numel() == 1:
            intercepts[seg] = TARGETS[target_name](x_seg)[0]

    slopes = _maybe_dyadic(slopes, config)
    intercepts = _maybe_dyadic(intercepts, config)
    return {
        "slopes": slopes,
        "intercepts": intercepts,
        "min_e_routing": min_e_routing,
        "max_e_routing": max_e_routing,
        "x_min": x_min,
        "x_max": x_max,
    }


def activation_prefix_index_torch(
    x: torch.Tensor,
    config: TorchBPLAConfig,
    min_e_routing: int,
    max_e_routing: int,
) -> torch.Tensor:
    x_clip = x.clamp(-float(config.activation_range), float(config.activation_range))
    fraction, exponent, sign = _fraction_and_exponent(x_clip)
    small_or_zero = (x_clip == 0) | (exponent < min_e_routing)
    exponent_bins = max_e_routing - min_e_routing + 1
    prefix = torch.clamp((fraction * (1 << config.prefix_bits)).floor().to(torch.long), 0, (1 << config.prefix_bits) - 1)
    exp_bin = exponent.clamp(min_e_routing, max_e_routing).to(torch.long) - min_e_routing
    sign_bin = sign.to(torch.long)
    idx = 1 + ((sign_bin * exponent_bins + exp_bin) << config.prefix_bits) + prefix
    return torch.where(small_or_zero, torch.zeros_like(idx), idx)


def bpla_activation_torch(x: torch.Tensor, table: dict[str, torch.Tensor | int | float], config: TorchBPLAConfig) -> torch.Tensor:
    idx = activation_prefix_index_torch(
        x,
        config,
        int(table["min_e_routing"]),
        int(table["max_e_routing"]),
    )
    x_clip = x.clamp(float(table["x_min"]), float(table["x_max"]))
    slopes = table["slopes"]
    intercepts = table["intercepts"]
    assert isinstance(slopes, torch.Tensor)
    assert isinstance(intercepts, torch.Tensor)
    return slopes[idx] * x_clip + intercepts[idx]


class TorchBPLALinear(nn.Module):
    def __init__(self, source: nn.Linear, config: TorchBPLAConfig, tables: SharedBPLATables | None = None):
        super().__init__()
        self.config = config
        self.tables = tables or SharedBPLATables(config)
        self.weight = nn.Parameter(source.weight.detach().clone(), requires_grad=False)
        if source.bias is None:
            self.bias = None
        else:
            self.bias = nn.Parameter(source.bias.detach().clone(), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return bpla_linear_torch(x, self.weight, self.bias, self.config, self.tables)


class TorchBPLAActivation(nn.Module):
    def __init__(
        self,
        target_name: str = "gelu",
        config: TorchBPLAConfig | None = None,
        tables: SharedBPLATables | None = None,
    ):
        super().__init__()
        self.target_name = target_name
        self.config = config or TorchBPLAConfig()
        self.tables = tables or SharedBPLATables(self.config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        table = self.tables.activation(self.target_name, x.device, x.dtype)
        return bpla_activation_torch(x, table, self.config)


class TorchBPLALayerNorm(nn.Module):
    """Drop-in inference proxy for a composed B-PLA LayerNorm."""

    def __init__(self, source: nn.LayerNorm, config: TorchBPLAConfig, tables: SharedBPLATables | None = None):
        super().__init__()
        self.normalized_shape = tuple(source.normalized_shape)
        self.eps = float(source.eps)
        self.config = config
        self.tables = tables or SharedBPLATables(config)
        self.weight = (
            nn.Parameter(source.weight.detach().clone(), requires_grad=False)
            if source.weight is not None
            else None
        )
        self.bias = (
            nn.Parameter(source.bias.detach().clone(), requires_grad=False)
            if source.bias is not None
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return bpla_layer_norm_torch(
            x,
            self.normalized_shape,
            self.weight,
            self.bias,
            self.eps,
            self.config,
            self.tables,
        )


def replace_layer_norms(
    module: nn.Module,
    config: TorchBPLAConfig,
    tables: SharedBPLATables | None = None,
) -> int:
    """Replace every ``nn.LayerNorm`` recursively with the B-PLA proxy."""

    shared = tables or SharedBPLATables(config)
    replaced = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.LayerNorm):
            setattr(module, name, TorchBPLALayerNorm(child, config, shared))
            replaced += 1
        else:
            replaced += replace_layer_norms(child, config, shared)
    module._bpla_layernorm_count = replaced
    return replaced


class TorchBPLAConv1D(nn.Module):
    """B-PLA proxy replacement for HuggingFace GPT-style Conv1D."""

    def __init__(self, source: Conv1D, config: TorchBPLAConfig, tables: SharedBPLATables | None = None):
        super().__init__()
        self.config = config
        self.tables = tables or SharedBPLATables(config)
        self.nf = source.nf
        self.weight = nn.Parameter(source.weight.detach().clone(), requires_grad=False)
        self.bias = nn.Parameter(source.bias.detach().clone(), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size_out = x.size()[:-1] + (self.nf,)
        out = bpla_linear_torch(x, self.weight.t(), self.bias, self.config, self.tables)
        return out.view(size_out)


def replace_linear_and_gelu(
    module: nn.Module,
    config: TorchBPLAConfig,
    replace_linear: bool = True,
    replace_gelu: bool = True,
    max_linear_modules: int | None = None,
    tables: SharedBPLATables | None = None,
) -> int:
    """In-place replacement helper for sensitivity checks on PyTorch models."""

    tables = tables or SharedBPLATables(config)
    if not tables._multiplier and not tables._activation:
        reference = next(module.parameters(), None)
        if reference is not None:
            if replace_linear:
                tables.multiplier(reference.device, reference.dtype)
            if replace_gelu:
                tables.activation("gelu", reference.device, reference.dtype)
    replaced_linear = 0
    for name, child in list(module.named_children()):
        if replace_linear and isinstance(child, nn.Linear) and (max_linear_modules is None or replaced_linear < max_linear_modules):
            setattr(module, name, TorchBPLALinear(child, config, tables))
            replaced_linear += 1
            continue
        child_name = child.__class__.__name__.lower()
        if replace_gelu and (isinstance(child, nn.GELU) or "gelu" in child_name):
            setattr(module, name, TorchBPLAActivation("gelu", config, tables))
            continue
        replaced_linear += replace_linear_and_gelu(
            child,
            config=config,
            replace_linear=replace_linear,
            replace_gelu=replace_gelu,
            max_linear_modules=None if max_linear_modules is None else max_linear_modules - replaced_linear,
            tables=tables,
        )
    return replaced_linear


def replace_gpt2_conv1d_and_gelu(
    module: nn.Module,
    config: TorchBPLAConfig,
    replace_conv1d: bool = True,
    replace_gelu: bool = True,
    max_conv1d_modules: int | None = None,
    tables: SharedBPLATables | None = None,
) -> int:
    """In-place replacement helper for GPT-2 style models."""

    tables = tables or SharedBPLATables(config)
    if not tables._multiplier and not tables._activation:
        reference = next(module.parameters(), None)
        if reference is not None:
            if replace_conv1d:
                tables.multiplier(reference.device, reference.dtype)
            if replace_gelu:
                tables.activation("gelu", reference.device, reference.dtype)
    replaced_conv = 0
    for name, child in list(module.named_children()):
        if replace_conv1d and isinstance(child, Conv1D) and (max_conv1d_modules is None or replaced_conv < max_conv1d_modules):
            setattr(module, name, TorchBPLAConv1D(child, config, tables))
            replaced_conv += 1
            continue
        if replace_gelu and isinstance(child, nn.GELU):
            setattr(module, name, TorchBPLAActivation("gelu", config, tables))
            continue
        if replace_gelu and child.__class__.__name__.lower().endswith("geluactivation"):
            setattr(module, name, TorchBPLAActivation("gelu", config, tables))
            continue
        replaced_conv += replace_gpt2_conv1d_and_gelu(
            child,
            config=config,
            replace_conv1d=replace_conv1d,
            replace_gelu=replace_gelu,
            max_conv1d_modules=None if max_conv1d_modules is None else max_conv1d_modules - replaced_conv,
            tables=tables,
        )
    return replaced_conv
