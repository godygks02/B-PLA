"""
Piecewise Affine Operations (PAO/PAM) baseline for direct comparison with B-PLA.

Reference implementation of the arithmetic defined in Kosson and Jaggi,
"Multiplication-Free Transformer Training via Piecewise Affine Operations",
NeurIPS 2023 (official code: https://github.com/epfml/pam).

Only the *forward* primitives are reproduced here. The original work also
replaces the backward pass and the optimizer; those are out of scope because
B-PLA targets training-free replacement in a pretrained model, so every
comparison in this file runs at inference time with zero weight updates.

Faithfulness notes
------------------
* ``pao_multiply_torch`` implements Eq. (5)-(8) of the paper exactly. It is
  verified in ``tests/test_pao.py`` against the equivalent Mogami trick of
  adding the float32 bit patterns as int32, which is what the official CUDA
  kernel does.
* ``pao_divide_torch`` implements Eq. (14)-(17). The published Eq. (17) prints
  ``+1{M_A - M_B <= 1}``, which is inconsistent with Eq. (16); we use the
  indicator on ``M_A - M_B < 0`` so the result mantissa stays in [0, 1), which
  is the only reading that makes division the inverse of Eq. (5)-(8).
* ``pao_gelu_torch`` is *not* from the paper. The original models use ReLU and
  the paper never defines a piecewise-affine GELU. We compose one from the PA
  primitives so that a nonlinear-path comparison against B-PLA is possible at
  all; it must be reported as our construction, not as a published baseline.

Relationship to B-PLA
---------------------
B-PLA writes the mantissa product as ``1 + m1 + m2 + m1*m2`` and approximates
the interaction term ``m1*m2`` with a prefix-indexed local plane. PAM is the
special case in which that plane is identically zero. PAM is therefore a strict
lower bound on B-PLA multiplier fidelity and an upper bound on its simplicity;
``tests/test_pao.py`` pins this relationship down numerically.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.pytorch_utils import Conv1D
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

try:
    from transformers import AttentionMaskInterface
    from transformers.masking_utils import eager_mask
except ImportError:  # Compatibility with pre-mask-registry Transformers.
    AttentionMaskInterface = None
    eager_mask = None

from modules.torch_bpla import ViTAttentionModule
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention


LOG2_E = 1.4426950408889634


@dataclass(frozen=True)
class TorchPAOConfig:
    """Configuration for the PAO forward baseline.

    ``alpha`` enables the optional single-constant error compensation sketched
    in Section 2.7 of the paper (``x1 * x2 * alpha``). The paper leaves this to
    future work and reports no results with it, so the default is ``None``.
    """

    matmul_chunk_out: int = 32
    alpha: float | None = None
    # Upper bound on the elements materialized by one chunk of the elementwise
    # broadcast. Emulating a scalar operation over a matmul is memory-bound, so
    # this matters more than the nominal chunk width.
    element_budget: int = int(os.environ.get("BPLA_MATMUL_ELEMENT_BUDGET", 8_000_000))


def _decompose(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split into (mantissa fraction M in [0,1), unbiased exponent E, sign bit)."""

    abs_x = x.abs()
    mant, exponent = torch.frexp(abs_x)
    fraction = torch.where(abs_x > 0, mant * 2.0 - 1.0, torch.zeros_like(abs_x))
    return fraction, exponent - 1, torch.signbit(x)


def _compose(fraction: torch.Tensor, exponent: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
    magnitude = torch.ldexp(1.0 + fraction, exponent)
    return torch.where(negative, -magnitude, magnitude)


#: See the matching note in torch_bpla: the elementwise chain is
#: memory-bound and fusing it removes most of the traffic.
_COMPILE = os.environ.get("BPLA_COMPILE", "0") not in {"0", "", "false", "False"}
_COMPILED: dict[str, Any] = {}


def _compiled(name: str, fn: Any) -> Any:
    """Return a fused version of ``fn``, or ``fn`` itself if fusing is unavailable.

    See the matching note in torch_bpla: a missing compiler toolchain must not
    change what the code computes.
    """

    if not _COMPILE:
        return fn
    if name not in _COMPILED:
        try:
            compiled = torch.compile(fn, dynamic=True)
            probe = torch.zeros(2)
            compiled(probe, probe)
            _COMPILED[name] = compiled
        except Exception as error:  # pragma: no cover - depends on the toolchain
            warnings.warn(
                f"BPLA_COMPILE was requested but {name} could not be fused "
                f"({type(error).__name__}); falling back to eager execution.",
                RuntimeWarning,
                stacklevel=2,
            )
            _COMPILED[name] = fn
    return _COMPILED[name]


def pao_multiply_torch(
    a: torch.Tensor,
    b: torch.Tensor,
    config: TorchPAOConfig | None = None,
) -> torch.Tensor:
    """Piecewise affine multiplication (PAM), Eq. (5)-(8)."""

    config = config or TorchPAOConfig()
    # The alpha branch recurses, so it stays outside the compiled region.
    result = _compiled("multiply", _pao_multiply_impl)(a, b)
    if config.alpha is not None:
        alpha = torch.full_like(result, config.alpha)
        result = _compiled("multiply", _pao_multiply_impl)(result, alpha)
    return result


def _pao_multiply_impl(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    dtype = torch.promote_types(a.dtype, b.dtype)
    a = a.to(dtype)
    b = b.to(dtype)

    frac_a, exp_a, sign_a = _decompose(a)
    frac_b, exp_b, sign_b = _decompose(b)

    mantissa_sum = frac_a + frac_b
    overflow = mantissa_sum >= 1.0
    fraction = mantissa_sum - overflow.to(dtype)
    exponent = exp_a + exp_b + overflow.to(exp_a.dtype)
    result = _compose(fraction, exponent, sign_a ^ sign_b)

    zero = (a == 0) | (b == 0)
    result = torch.where(zero, torch.zeros_like(result), result)
    # NaN/Inf are handled explicitly in the paper; deferring to exact IEEE
    # semantics reproduces the same propagation without special-casing.
    special = ~(torch.isfinite(a) & torch.isfinite(b))
    if special.any():
        result = torch.where(special, a * b, result)

    return result


def pao_divide_torch(
    a: torch.Tensor,
    b: torch.Tensor,
    config: TorchPAOConfig | None = None,
) -> torch.Tensor:
    """Piecewise affine division, Eq. (14)-(17)."""

    del config
    dtype = torch.promote_types(a.dtype, b.dtype)
    a = a.to(dtype)
    b = b.to(dtype)

    frac_a, exp_a, sign_a = _decompose(a)
    frac_b, exp_b, sign_b = _decompose(b)

    mantissa_diff = frac_a - frac_b
    underflow = mantissa_diff < 0.0
    fraction = mantissa_diff + underflow.to(dtype)
    exponent = exp_a - exp_b - underflow.to(exp_a.dtype)
    result = _compose(fraction, exponent, sign_a ^ sign_b)

    result = torch.where(a == 0, torch.zeros_like(result), result)
    degenerate = (b == 0) | ~(torch.isfinite(a) & torch.isfinite(b))
    if degenerate.any():
        result = torch.where(degenerate, a / b, result)
    return result


def paexp2_torch(x: torch.Tensor) -> torch.Tensor:
    """``paexp2(A) = 2^floor(A) * (1 + A - floor(A))``, Eq. (9).

    Special values need explicit handling, as the paper states for its own
    implementation. Without it ``x - floor(x)`` is ``inf - inf`` at an infinite
    input and the whole expression returns NaN -- which poisons any masked
    softmax, since an additive causal mask supplies exactly that input.
    """

    floor = torch.floor(x)
    fraction = x - floor
    exponent = torch.nan_to_num(floor, nan=0.0, posinf=128.0, neginf=-149.0)
    result = torch.ldexp(1.0 + fraction, exponent.to(torch.int32).clamp(-149, 128))

    # Underflow and overflow the way the exact function does, rather than
    # saturating at the smallest subnormal.
    result = torch.where(floor < -149, torch.zeros_like(result), result)
    result = torch.where(floor > 128, torch.full_like(result, float("inf")), result)
    result = torch.where(torch.isneginf(x), torch.zeros_like(result), result)
    result = torch.where(torch.isposinf(x), torch.full_like(result, float("inf")), result)
    return torch.where(torch.isnan(x), torch.full_like(result, float("nan")), result)


def palog2_torch(x: torch.Tensor) -> torch.Tensor:
    """``palog2(A) = E_A + M_A`` for A > 0, Eq. (10)."""

    fraction, exponent, _ = _decompose(x)
    result = exponent.to(x.dtype) + fraction
    result = torch.where(x > 0, result, torch.full_like(result, float("-inf")))
    return torch.where(x < 0, torch.full_like(result, float("nan")), result)


def paexp_torch(x: torch.Tensor, config: TorchPAOConfig | None = None) -> torch.Tensor:
    """``paexp(A) = paexp2(log2(e) * A)``, Eq. (18)."""

    scale = torch.full_like(x, LOG2_E)
    return paexp2_torch(pao_multiply_torch(scale, x, config))


def palog_torch(x: torch.Tensor, config: TorchPAOConfig | None = None) -> torch.Tensor:
    """``palog(A) = palog2(A) / log2(e)``, Eq. (19)."""

    return pao_divide_torch(palog2_torch(x), torch.full_like(x, LOG2_E), config)


def pasqrt_torch(x: torch.Tensor, config: TorchPAOConfig | None = None) -> torch.Tensor:
    """``pasqrt(A) = paexp2(palog2(A) / 2)``, Eq. (20)."""

    return paexp2_torch(pao_divide_torch(palog2_torch(x), torch.full_like(x, 2.0), config))


def pao_linear_torch(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    config: TorchPAOConfig | None = None,
) -> torch.Tensor:
    """Linear layer whose scalar products are PAM, chunked over output units."""

    config = config or TorchPAOConfig()
    original_shape = x.shape[:-1]
    x_flat = x.reshape(-1, x.shape[-1])
    rows = []
    chunk = max(1, min(config.matmul_chunk_out, config.element_budget // max(1, x_flat.numel())))
    for start in range(0, weight.shape[0], chunk):
        w = weight[start : start + chunk]
        products = pao_multiply_torch(x_flat[:, None, :], w[None, :, :], config)
        out = products.sum(dim=-1)
        if bias is not None:
            out = out + bias[start : start + chunk]
        rows.append(out)
    return torch.cat(rows, dim=-1).reshape(*original_shape, weight.shape[0])


def _elementwise_chunk(a: torch.Tensor, b: torch.Tensor, requested: int, budget: int) -> int:
    """Cap the output chunk so the broadcast product stays within ``budget``.

    Chunking over the output dimension alone is not enough: attention's PV
    matmul has only 64 output columns, so a chunk of 128 does not chunk at all
    and the elementwise broadcast materializes batch*heads*M*N*K values at once.
    Sizing the chunk by the actual element count keeps every operation bounded
    regardless of shape. This changes memory and speed only -- the result is
    identical, which ``tests/test_pao.py`` pins down.
    """

    leading = 1
    for dimension in torch.broadcast_shapes(a.shape[:-2], b.shape[:-2]):
        leading *= dimension
    per_column = max(1, leading * a.shape[-2] * a.shape[-1])
    return max(1, min(requested, budget // per_column))


def pao_matmul_torch(
    a: torch.Tensor,
    b: torch.Tensor,
    config: TorchPAOConfig | None = None,
) -> torch.Tensor:
    """Batched PAM matmul for ``[..., M, K] @ [..., K, N]``."""

    config = config or TorchPAOConfig()
    if a.ndim < 2 or b.ndim < 2:
        raise ValueError("PAO matmul inputs must have at least two dimensions.")
    if a.shape[-1] != b.shape[-2]:
        raise ValueError(f"Incompatible PAO matmul shapes: {tuple(a.shape)} and {tuple(b.shape)}")

    outputs = []
    chunk = _elementwise_chunk(a, b, config.matmul_chunk_out, config.element_budget)
    for start in range(0, b.shape[-1], chunk):
        b_chunk = b[..., :, start : start + chunk]
        products = pao_multiply_torch(
            a.unsqueeze(-2),
            b_chunk.transpose(-1, -2).unsqueeze(-3),
            config,
        )
        outputs.append(products.sum(dim=-1))
    return torch.cat(outputs, dim=-1)


def pao_softmax_torch(
    x: torch.Tensor,
    dim: int = -1,
    config: TorchPAOConfig | None = None,
) -> torch.Tensor:
    """Softmax composed from ``paexp`` and piecewise affine division."""

    shifted = x - x.amax(dim=dim, keepdim=True)
    exponentials = paexp_torch(shifted, config)
    denominator = exponentials.sum(dim=dim, keepdim=True)
    return pao_divide_torch(exponentials, denominator.expand_as(exponentials), config)


def pao_layer_norm_torch(
    x: torch.Tensor,
    normalized_shape: tuple[int, ...],
    weight: torch.Tensor | None,
    bias: torch.Tensor | None,
    eps: float,
    config: TorchPAOConfig | None = None,
) -> torch.Tensor:
    """LayerNorm composed from PAM, piecewise affine division and ``pasqrt``."""

    dims = tuple(range(x.ndim - len(normalized_shape), x.ndim))
    count = torch.tensor(
        float(torch.Size(normalized_shape).numel()), device=x.device, dtype=x.dtype
    )
    mean = pao_divide_torch(x.sum(dim=dims, keepdim=True), count.expand(1), config)
    centered = x - mean
    squared = pao_multiply_torch(centered, centered, config)
    variance = pao_divide_torch(squared.sum(dim=dims, keepdim=True), count.expand(1), config)
    deviation = pasqrt_torch(variance + eps, config)
    normalized = pao_divide_torch(centered, deviation.expand_as(centered), config)
    if weight is not None:
        normalized = pao_multiply_torch(normalized, weight, config)
    if bias is not None:
        normalized = normalized + bias
    return normalized


def pao_gelu_torch(x: torch.Tensor, config: TorchPAOConfig | None = None) -> torch.Tensor:
    """GELU composed from PA primitives via the sigmoid form ``x * sigma(1.702x)``.

    Not defined in the original paper (its models use ReLU); this is our
    construction so that a nonlinear-path comparison is possible.
    """

    scaled = pao_multiply_torch(torch.full_like(x, 1.702), x, config)
    exponential = paexp_torch(-scaled, config)
    sigmoid = pao_divide_torch(torch.ones_like(x), 1.0 + exponential, config)
    return pao_multiply_torch(x, sigmoid, config)


class TorchPAOLinear(nn.Module):
    _span_coverage_kinds = frozenset({"multiply", "mac"})

    def __init__(self, source: nn.Linear, config: TorchPAOConfig, share_weight: bool = False):
        super().__init__()
        self.config = config
        # See TorchBPLALinear: sharing preserves GPT-2's lm_head/embedding tie.
        self.weight = (
            source.weight
            if share_weight
            else nn.Parameter(source.weight.detach().clone(), requires_grad=False)
        )
        self.bias = (
            None
            if source.bias is None
            else nn.Parameter(source.bias.detach().clone(), requires_grad=False)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return pao_linear_torch(x, self.weight, self.bias, self.config)


class TorchPAOConv2d(nn.Module):
    """PAO proxy for a 2-D convolution, mirroring ``TorchBPLAConv2d``."""

    _span_coverage_kinds = frozenset({"multiply", "mac"})

    def __init__(self, source: nn.Conv2d, config: TorchPAOConfig):
        super().__init__()
        if source.groups != 1:
            raise NotImplementedError("Grouped convolutions are not supported by the PAO proxy.")
        self.config = config
        self.kernel_size = tuple(source.kernel_size)
        self.stride = tuple(source.stride)
        self.padding = tuple(source.padding) if isinstance(source.padding, tuple) else source.padding
        self.dilation = tuple(source.dilation)
        self.out_channels = source.out_channels
        self.weight = nn.Parameter(source.weight.detach().clone(), requires_grad=False)
        self.bias = (
            None
            if source.bias is None
            else nn.Parameter(source.bias.detach().clone(), requires_grad=False)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        patches = F.unfold(
            x,
            kernel_size=self.kernel_size,
            dilation=self.dilation,
            padding=self.padding,
            stride=self.stride,
        )
        out = pao_linear_torch(
            patches.transpose(1, 2),
            self.weight.reshape(self.out_channels, -1),
            self.bias,
            self.config,
        )
        height, width = x.shape[-2], x.shape[-1]
        pad_h, pad_w = (self.padding, self.padding) if isinstance(self.padding, int) else self.padding
        out_h = (height + 2 * pad_h - self.dilation[0] * (self.kernel_size[0] - 1) - 1) // self.stride[0] + 1
        out_w = (width + 2 * pad_w - self.dilation[1] * (self.kernel_size[1] - 1) - 1) // self.stride[1] + 1
        return out.transpose(1, 2).reshape(x.shape[0], self.out_channels, out_h, out_w)


class TorchPAOConv1D(nn.Module):
    _span_coverage_kinds = frozenset({"multiply", "mac"})

    def __init__(self, source: Conv1D, config: TorchPAOConfig):
        super().__init__()
        self.config = config
        self.nf = source.nf
        self.weight = nn.Parameter(source.weight.detach().clone(), requires_grad=False)
        self.bias = nn.Parameter(source.bias.detach().clone(), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size_out = x.size()[:-1] + (self.nf,)
        out = pao_linear_torch(x, self.weight.t(), self.bias, self.config)
        return out.view(size_out)


class TorchPAOActivation(nn.Module):
    _span_coverage_kinds = frozenset({"transcendental"})

    def __init__(self, config: TorchPAOConfig | None = None):
        super().__init__()
        self.config = config or TorchPAOConfig()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return pao_gelu_torch(x, self.config)


class TorchPAOLayerNorm(nn.Module):
    _span_coverage_kinds = frozenset({"multiply", "transcendental", "normalization"})

    def __init__(self, source: nn.LayerNorm, config: TorchPAOConfig):
        super().__init__()
        self.normalized_shape = tuple(source.normalized_shape)
        self.eps = float(source.eps)
        self.config = config
        self.weight = (
            None
            if source.weight is None
            else nn.Parameter(source.weight.detach().clone(), requires_grad=False)
        )
        self.bias = (
            None
            if source.bias is None
            else nn.Parameter(source.bias.detach().clone(), requires_grad=False)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return pao_layer_norm_torch(
            x, self.normalized_shape, self.weight, self.bias, self.eps, self.config
        )


def replace_pao_linear_and_gelu(
    module: nn.Module,
    config: TorchPAOConfig,
    replace_linear: bool = True,
    replace_gelu: bool = True,
    max_linear_modules: int | None = None,
    replace_conv2d: bool = False,
) -> int:
    """In-place PAO replacement mirroring ``replace_linear_and_gelu``."""

    replaced_linear = 0
    for name, child in list(module.named_children()):
        if (
            replace_linear
            and isinstance(child, nn.Linear)
            and (max_linear_modules is None or replaced_linear < max_linear_modules)
        ):
            setattr(module, name, TorchPAOLinear(child, config))
            replaced_linear += 1
            continue
        if replace_conv2d and isinstance(child, nn.Conv2d):
            setattr(module, name, TorchPAOConv2d(child, config))
            replaced_linear += 1
            continue
        child_name = child.__class__.__name__.lower()
        if replace_gelu and (isinstance(child, nn.GELU) or "gelu" in child_name):
            setattr(module, name, TorchPAOActivation(config))
            continue
        replaced_linear += replace_pao_linear_and_gelu(
            child,
            config=config,
            replace_linear=replace_linear,
            replace_gelu=replace_gelu,
            max_linear_modules=None
            if max_linear_modules is None
            else max_linear_modules - replaced_linear,
            replace_conv2d=replace_conv2d,
        )
    return replaced_linear


def replace_pao_gpt2_conv1d_and_gelu(
    module: nn.Module,
    config: TorchPAOConfig,
    replace_conv1d: bool = True,
    replace_gelu: bool = True,
    max_conv1d_modules: int | None = None,
    replace_lm_head: bool = False,
) -> int:
    """In-place PAO replacement mirroring ``replace_gpt2_conv1d_and_gelu``."""

    replaced_conv = 0
    for name, child in list(module.named_children()):
        if (
            replace_conv1d
            and isinstance(child, Conv1D)
            and (max_conv1d_modules is None or replaced_conv < max_conv1d_modules)
        ):
            setattr(module, name, TorchPAOConv1D(child, config))
            replaced_conv += 1
            continue
        if replace_lm_head and isinstance(child, nn.Linear):
            setattr(module, name, TorchPAOLinear(child, config, share_weight=True))
            replaced_conv += 1
            continue
        child_name = child.__class__.__name__.lower()
        if replace_gelu and (isinstance(child, nn.GELU) or child_name.endswith("geluactivation")):
            setattr(module, name, TorchPAOActivation(config))
            continue
        replaced_conv += replace_pao_gpt2_conv1d_and_gelu(
            child,
            config=config,
            replace_conv1d=replace_conv1d,
            replace_gelu=replace_gelu,
            max_conv1d_modules=None
            if max_conv1d_modules is None
            else max_conv1d_modules - replaced_conv,
            replace_lm_head=replace_lm_head,
        )
    return replaced_conv


def replace_pao_layer_norms(module: nn.Module, config: TorchPAOConfig) -> int:
    replaced = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.LayerNorm):
            setattr(module, name, TorchPAOLayerNorm(child, config))
            replaced += 1
        else:
            replaced += replace_pao_layer_norms(child, config)
    module._pao_layernorm_count = replaced
    return replaced


def replace_pao_attention_matmuls(
    module: nn.Module,
    config: TorchPAOConfig,
    mode: str = "pao-full",
    approximate_softmax: bool = False,
) -> int:
    """Install a PAO attention path, mirroring ``replace_attention_matmuls``."""

    valid_modes = {"exact", "pao-qk", "pao-pv", "pao-full"}
    if mode not in valid_modes:
        raise ValueError(f"Unknown attention mode {mode!r}. Choose one of {sorted(valid_modes)}.")

    attention_modules = [
        child for child in module.modules() if isinstance(child, (ViTAttentionModule, GPT2Attention))
    ]
    if not attention_modules:
        return 0

    interface_name = f"pao_{id(config)}_{mode}"

    def pao_attention_forward(
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

        use_qk = mode in {"pao-qk", "pao-full"}
        use_pv = mode in {"pao-pv", "pao-full"}
        if use_qk:
            attention_scores = pao_matmul_torch(query, key.transpose(-1, -2), config)
        else:
            attention_scores = torch.matmul(query, key.transpose(-1, -2))
        attention_scores = attention_scores * scaling

        attention_weights = attention_scores
        if attention_mask is not None:
            attention_weights = attention_weights + attention_mask
        if approximate_softmax:
            attention_weights = pao_softmax_torch(attention_weights, dim=-1, config=config)
        else:
            attention_weights = F.softmax(attention_weights, dim=-1)
        attention_weights = attention_weights.type(value.dtype)
        attention_weights = F.dropout(
            attention_weights, p=dropout, training=attention_module.training
        )
        if use_pv:
            attention_output = pao_matmul_torch(attention_weights, value, config)
        else:
            attention_output = torch.matmul(attention_weights, value)
        return attention_output.transpose(1, 2), attention_weights

    ALL_ATTENTION_FUNCTIONS.register(interface_name, pao_attention_forward)
    if AttentionMaskInterface is not None and eager_mask is not None:
        AttentionMaskInterface.register(interface_name, eager_mask)
    for attention_module in attention_modules:
        attention_module.config._attn_implementation = interface_name
    module._pao_attention_mode = mode
    module._pao_softmax_enabled = approximate_softmax
    return len(attention_modules)
