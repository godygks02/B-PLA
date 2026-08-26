"""
W8A8 post-training quantization baseline for direct comparison with B-PLA.

Post-training quantization is the de-facto standard for the training-free
setting this project targets, so it -- not PAO -- is the baseline a reviewer
will reach for first. This module inserts a standard W8A8 PTQ path into the
same pretrained checkpoint under the same harness as ``torch_pao`` and
``torch_bpla``, with zero weight updates in every condition.

What "standard" means here
--------------------------
The configuration defaults are the ones that make PTQ *strong*, because a
baseline that loses only because it was built badly proves nothing:

* **per-output-channel symmetric weight quantization** (``s_c = max|w_c| / 127``).
  Per-tensor weight scales are a well-known way to make W8A8 look worse than it
  is; they are available via ``per_channel_weights=False`` only so the test
  suite can show the difference.
* **percentile activation calibration** over a growing histogram, rather than
  min-max. A single outlier activation otherwise sets the scale for every value
  in the tensor, which is the classic failure mode of naive W8A8 on
  transformers.
* **unsigned quantization of the attention probabilities**, which live in
  [0, 1]. Spending a sign bit on a non-negative tensor throws away one of the
  eight bits for nothing.

Faithfulness notes
------------------
* Quantization is *simulated*, not executed on integer kernels: every tensor is
  quantized and dequantized, then the product is taken in float32. The values
  are those an int8 multiplier would produce; only the accumulator differs, and
  it differs in the direction that matters least -- a real int32 accumulator is
  exact, while float32 accumulation over K=3072 terms costs a relative error
  around 3e-6, roughly three orders of magnitude below the ~1e-3 that
  quantization itself introduces. Using float32 also keeps the accumulator
  *identical* to the exact, PAO and B-PLA backends, so the comparison isolates
  the multiplier rather than mixing in an accumulator change.
* Only scale groupings a real int8 GEMM can factor out of the accumulation are
  offered. Per-token activation scales and per-output-channel weight scales both
  factor out of ``sum_k x[m,k] w[n,k]``; a per-position scale on the attention
  ``value`` tensor would not, so the attention matmuls stay per-tensor, which is
  also the usual choice for int8 attention.
* The ``1/sqrt(head_dim)`` attention scaling is left in float32. Real int8
  attention folds it into the dequantization scale of the int32 accumulator, so
  quantizing it separately would model a multiplier that no implementation uses.
  This differs from the PAO and B-PLA paths, which do route that scaling through
  their approximate multipliers, and the difference is in PTQ's favour.
* Nonlinear paths (GELU, Softmax, LayerNorm) are deliberately left exact. That
  is the PTQ convention, and it is why this backend is only meaningful at the
  ``multiplication`` scope; see ``replace_ptq_*`` and the harness, which refuse
  the other scopes rather than reporting a mislabelled row.

Relationship to B-PLA
---------------------
Both methods remove float multiplication from a pretrained model without
retraining, but they remove different things. PTQ replaces a float multiplier
with a *narrower* multiplier: int8xint8 hardware is still a multiplier array.
B-PLA replaces the multiplication with shift-add. The comparison axis is
therefore not accuracy alone but accuracy at a stated arithmetic cost.

Calibration
-----------
``calibrate_ptq_model`` runs the *float* model forward over calibration batches
and records activation histograms; it never sees labels and updates no weights,
which is the same contract B-PLA's ``calibrate_model_activation_range`` works
under, so both backends may be given the same data and the same sample count.

Until ``finalize_ptq_model`` has been called, every quantized module raises on
forward. An uncalibrated PTQ model that quietly returned exact results would be
the worst possible failure here: it would look like a flawless baseline.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Iterable

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


#: Floor for any quantization scale. A weight row or an activation tensor that
#: is identically zero would otherwise divide by zero and produce NaN where the
#: correct answer is simply zero.
_TINY = 1e-12

UNCALIBRATED = "uncalibrated"
OBSERVING = "observing"
QUANTIZED = "quantized"

GRANULARITIES = ("tensor", "token")


@dataclass(frozen=True)
class TorchPTQConfig:
    """Configuration for the W8A8 PTQ baseline.

    ``activation_granularity`` selects between static per-tensor activation
    scales, the conventional W8A8 setting, and dynamic per-token scales, which
    are computed at run time from the row being quantized. Both are realizable
    by an int8 GEMM: a per-row scale on the activation factors out of the
    accumulation exactly as a per-output-channel weight scale does. Per-token
    scales apply to the weighted matmuls only; see the module docstring for why
    attention stays per-tensor.

    ``activation_percentile`` applies to the ``tensor`` setting only. Per-token
    scaling is min-max over the row, which is what every dynamic-quantization
    implementation does and is also the better choice: a percentile over a
    single row of 768 values is mostly noise. Measured on GPT-2, clipping is
    actively harmful either way -- see the note in ``experiments/`` -- so the
    percentile is a knob for reproducing the conventional recipe, not a
    recommendation.
    """

    weight_bits: int = 8
    activation_bits: int = 8
    per_channel_weights: bool = True
    activation_percentile: float = 99.99
    histogram_bins: int = 2048
    activation_granularity: str = "tensor"
    #: When set, attention probabilities go on a power-of-two grid at this bit
    #: width instead of the uniform activation grid -- FQ-ViT's Log-Int-Softmax,
    #: which the paper runs at 4 bits. Softmax output is bounded to (0, 1), so
    #: unlike every other activation here it needs no calibrated range.
    softmax_log2_bits: int | None = None


def _validate_config(config: TorchPTQConfig) -> None:
    if not 2 <= config.weight_bits <= 16:
        raise ValueError(f"weight_bits must be in [2, 16], got {config.weight_bits}.")
    if not 2 <= config.activation_bits <= 16:
        raise ValueError(f"activation_bits must be in [2, 16], got {config.activation_bits}.")
    if not 0.0 < config.activation_percentile <= 100.0:
        raise ValueError(
            f"activation_percentile must be in (0, 100], got {config.activation_percentile}."
        )
    if config.histogram_bins < 16:
        raise ValueError(f"histogram_bins must be at least 16, got {config.histogram_bins}.")
    if config.softmax_log2_bits is not None and not 2 <= config.softmax_log2_bits <= 8:
        raise ValueError(
            f"softmax_log2_bits must be in [2, 8], got {config.softmax_log2_bits}."
        )
    if config.activation_granularity not in GRANULARITIES:
        raise ValueError(
            f"Unknown activation_granularity {config.activation_granularity!r}. "
            f"Choose one of {list(GRANULARITIES)}."
        )


# ------------------------------------------------------------------- primitives


def quantize_weight_per_channel(
    weight: torch.Tensor,
    bits: int = 8,
    per_channel: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric weight quantization; returns ``(dequantized, scale)``.

    Channels are the leading dimension, which is the output unit for
    ``nn.Linear`` ([out, in]) and for ``nn.Conv2d`` ([out, in, kh, kw]) alike.
    GPT-2's ``Conv1D`` stores the transpose and is handled by transposing before
    the call rather than by adding an axis argument here.
    """

    qmax = float(2 ** (bits - 1) - 1)
    if per_channel and weight.ndim > 1:
        reduce_dims = tuple(range(1, weight.ndim))
        amax = weight.detach().abs().amax(dim=reduce_dims, keepdim=True)
    else:
        amax = weight.detach().abs().amax().reshape(*([1] * weight.ndim))
    scale = (amax / qmax).clamp(min=_TINY)
    quantized = torch.clamp(torch.round(weight.detach() / scale), -qmax, qmax)
    return quantized * scale, scale


def fake_quantize_activation(
    x: torch.Tensor,
    amax: torch.Tensor | float,
    bits: int = 8,
    signed: bool = True,
) -> torch.Tensor:
    """Quantize to ``bits`` and immediately dequantize.

    ``signed`` selects the symmetric [-qmax, qmax] grid used for anything that
    can be negative; the unsigned [0, 2^bits - 1] grid is for tensors known to
    be non-negative, where a sign bit would be wasted. Attention probabilities
    are the case that matters.
    """

    if not isinstance(amax, torch.Tensor):
        amax = torch.tensor(float(amax), device=x.device, dtype=x.dtype)
    amax = amax.to(device=x.device, dtype=x.dtype).clamp(min=_TINY)
    if signed:
        qmax = float(2 ** (bits - 1) - 1)
        scale = amax / qmax
        return torch.clamp(torch.round(x / scale), -qmax, qmax) * scale
    qmax = float(2**bits - 1)
    scale = amax / qmax
    return torch.clamp(torch.round(x / scale), 0.0, qmax) * scale


class ActivationObserver:
    """Percentile range observer backed by a histogram that grows on demand.

    A percentile needs the distribution, not just the extremes, and storing the
    activations to sort them is not an option at these tensor sizes. The
    histogram is built over ``|x|`` on whatever device the activations live on.
    When a later batch exceeds the current upper bound the bin width is
    multiplied by an integer factor so that existing bins merge into the new
    ones exactly: every old bin falls wholly inside one new bin, so no count is
    lost or redistributed. The grid's upper edge does still depend on the batch
    order -- growing from a small first batch overshoots the true maximum by up
    to one factor -- so the resolved range is order-independent only to within
    one bin width, which at the default 2048 bins is 0.05% of the range.
    """

    def __init__(
        self,
        bits: int = 8,
        percentile: float = 99.99,
        bins: int = 2048,
        signed: bool = True,
    ):
        self.bits = bits
        self.percentile = percentile
        self.bins = bins
        self.signed = signed
        self.upper = 0.0
        self.observed_max = 0.0
        self.count = 0
        self._histogram: torch.Tensor | None = None

    @property
    def calibrated(self) -> bool:
        return self._histogram is not None

    @torch.no_grad()
    def observe(self, x: torch.Tensor) -> None:
        values = x.detach().abs().reshape(-1).float()
        if values.numel() == 0:
            return
        batch_max = float(values.amax())
        self.observed_max = max(self.observed_max, batch_max)
        self.count += values.numel()
        if self._histogram is None:
            self.upper = max(batch_max, _TINY)
            self._histogram = torch.zeros(self.bins, dtype=torch.float64, device=values.device)
        elif batch_max > self.upper:
            self._grow(batch_max)
        self._histogram += torch.histc(
            values, bins=self.bins, min=0.0, max=self.upper
        ).double()

    def _grow(self, new_upper: float) -> None:
        assert self._histogram is not None
        # An integer factor keeps the merge exact: new bin j is precisely the
        # union of old bins [j*factor, (j+1)*factor).
        factor = max(2, int(math.ceil(new_upper / max(self.upper, _TINY))))
        padded = torch.zeros(self.bins * factor, dtype=torch.float64, device=self._histogram.device)
        padded[: self.bins] = self._histogram
        self._histogram = padded.reshape(self.bins, factor).sum(dim=1)
        self.upper *= factor

    def resolve(self) -> float:
        """Return the clipping range: the upper edge of the percentile bin."""

        if self._histogram is None:
            raise RuntimeError(
                "ActivationObserver.resolve() was called before any activation was "
                "observed. Run calibrate_ptq_model() first."
            )
        if self.percentile >= 100.0:
            return max(self.observed_max, _TINY)
        total = float(self._histogram.sum())
        if total <= 0.0:
            return max(self.observed_max, _TINY)
        cumulative = torch.cumsum(self._histogram, dim=0)
        threshold = torch.tensor(
            [total * (self.percentile / 100.0)], dtype=torch.float64, device=cumulative.device
        )
        index = min(int(torch.searchsorted(cumulative, threshold)[0]), self.bins - 1)
        return max((index + 1) * self.upper / self.bins, _TINY)


# ---------------------------------------------------------------------- modules


class _PTQQuantizedModule(nn.Module):
    """Shared calibration state machine for the weighted-matmul replacements.

    Three states, and the initial one refuses to run. A PTQ model that was never
    calibrated but still produced numbers would produce *exact* numbers, and an
    exact baseline reported as a quantized one is a silent, plausible-looking
    lie -- exactly the failure this project has already paid for once by reading
    speed instead of accuracy.
    """

    def __init__(self, config: TorchPTQConfig):
        super().__init__()
        _validate_config(config)
        self.config = config
        self.ptq_state = UNCALIBRATED
        # Every weighted-layer input in these models is signed: hidden states,
        # normalized pixels, and even the post-GELU tensor, which reaches
        # -0.17. There is no unsigned case here, unlike attention.
        self.observer = ActivationObserver(
            bits=config.activation_bits,
            percentile=config.activation_percentile,
            bins=config.histogram_bins,
        )
        self.register_buffer("input_amax", torch.zeros((), dtype=torch.float32))

    def _prepare_input(self, x: torch.Tensor) -> torch.Tensor:
        if self.ptq_state == UNCALIBRATED:
            raise RuntimeError(
                f"{type(self).__name__} was used before calibration. Call "
                "calibrate_ptq_model(...) and then finalize_ptq_model(...)."
            )
        if self.ptq_state == OBSERVING:
            self.observer.observe(x)
            return x
        if self.config.activation_granularity == "token":
            # A per-row scale factors out of sum_k x[m,k] w[n,k], so an int8
            # GEMM can realize this; it is computed from the tensor at hand and
            # needs no calibrated range.
            amax = x.detach().abs().amax(dim=-1, keepdim=True)
        else:
            amax = self.input_amax
        return fake_quantize_activation(x, amax, self.config.activation_bits, signed=True)

    def _finalize(self) -> float:
        if self.ptq_state == QUANTIZED:
            return float(self.input_amax)
        amax = self.observer.resolve()
        self.input_amax = torch.tensor(
            amax, dtype=torch.float32, device=self.input_amax.device
        )
        self._quantize_weight()
        self.ptq_state = QUANTIZED
        return amax

    def _quantize_weight(self) -> None:
        """Replace the stored float weight with its dequantized int8 image.

        Deferred to finalization rather than done at construction so that
        calibration observes the activations of the genuinely float model, which
        is what every PTQ toolkit calibrates on.
        """

        weight, scale = quantize_weight_per_channel(
            self._weight_for_quantization(),
            self.config.weight_bits,
            self.config.per_channel_weights,
        )
        self._store_quantized_weight(weight)
        self.register_buffer("weight_scale", scale.detach().clone())

    def _weight_for_quantization(self) -> torch.Tensor:
        return self.weight

    def _store_quantized_weight(self, weight: torch.Tensor) -> None:
        self.weight = nn.Parameter(weight, requires_grad=False)


class TorchPTQLinear(_PTQQuantizedModule):
    """W8A8 replacement for ``nn.Linear``, mirroring ``TorchBPLALinear``."""

    _span_coverage_kinds = frozenset({"multiply", "mac"})

    def __init__(self, source: nn.Linear, config: TorchPTQConfig):
        super().__init__(config)
        # Unlike the B-PLA and PAO proxies there is no ``share_weight`` option:
        # quantizing the weight necessarily materializes a new tensor, so
        # GPT-2's lm_head/embedding tie cannot survive here. Nothing trains, so
        # the tie has no effect on the result; it only costs the memory saving.
        self.weight = nn.Parameter(source.weight.detach().clone(), requires_grad=False)
        self.bias = (
            None
            if source.bias is None
            else nn.Parameter(source.bias.detach().clone(), requires_grad=False)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(self._prepare_input(x), self.weight, self.bias)


class TorchPTQConv1D(_PTQQuantizedModule):
    """W8A8 replacement for HuggingFace GPT-style ``Conv1D``."""

    _span_coverage_kinds = frozenset({"multiply", "mac"})

    def __init__(self, source: Conv1D, config: TorchPTQConfig):
        super().__init__(config)
        self.nf = source.nf
        # Conv1D stores [in, out]; per-output-channel means per column.
        self.weight = nn.Parameter(source.weight.detach().clone(), requires_grad=False)
        self.bias = nn.Parameter(source.bias.detach().clone(), requires_grad=False)

    def _weight_for_quantization(self) -> torch.Tensor:
        return self.weight.t().contiguous()

    def _store_quantized_weight(self, weight: torch.Tensor) -> None:
        self.weight = nn.Parameter(weight.t().contiguous(), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size_out = x.size()[:-1] + (self.nf,)
        out = F.linear(self._prepare_input(x), self.weight.t(), self.bias)
        return out.view(size_out)


class TorchPTQConv2d(_PTQQuantizedModule):
    """W8A8 replacement for ``nn.Conv2d``, mirroring ``TorchBPLAConv2d``.

    The B-PLA proxy unfolds into a matmul because it has to emulate the scalar
    product; here the convolution can stay a convolution, since quantizing the
    operands and calling ``F.conv2d`` computes the same products an int8
    convolution kernel would.
    """

    _span_coverage_kinds = frozenset({"multiply", "mac"})

    def __init__(self, source: nn.Conv2d, config: TorchPTQConfig):
        super().__init__(config)
        if source.groups != 1:
            raise NotImplementedError("Grouped convolutions are not supported by the PTQ proxy.")
        self.stride = tuple(source.stride)
        self.padding = tuple(source.padding) if isinstance(source.padding, tuple) else source.padding
        self.dilation = tuple(source.dilation)
        self.weight = nn.Parameter(source.weight.detach().clone(), requires_grad=False)
        self.bias = (
            None
            if source.bias is None
            else nn.Parameter(source.bias.detach().clone(), requires_grad=False)
        )

    def _prepare_input(self, x: torch.Tensor) -> torch.Tensor:
        # A convolution contracts over channel *and* kernel position, so a
        # per-row scale on the flattened patch is not what the NCHW layout
        # offers; per-tensor is the realizable choice here either way.
        if self.ptq_state == QUANTIZED and self.config.activation_granularity == "token":
            return fake_quantize_activation(
                x, self.input_amax, self.config.activation_bits, signed=True
            )
        return super()._prepare_input(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(
            self._prepare_input(x),
            self.weight,
            self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
        )


# -------------------------------------------------------------------- attention


class PTQAttentionObservers:
    """Per-layer activation observers for the two activation-activation matmuls.

    Attention is where PTQ differs most from a weighted layer: both operands of
    ``QK^T`` and of ``PV`` are activations, so neither has a weight scale that
    can be computed offline. Each attention layer gets its own observers, since
    the score and probability distributions differ substantially by depth.
    """

    def __init__(self, config: TorchPTQConfig):
        _validate_config(config)
        self.config = config
        self.state = UNCALIBRATED
        self.observers = {
            "query": ActivationObserver(config.activation_bits, config.activation_percentile, config.histogram_bins),
            "key": ActivationObserver(config.activation_bits, config.activation_percentile, config.histogram_bins),
            # Softmax output is non-negative, so it gets the full 8 bits of
            # magnitude instead of 7 plus an unused sign.
            "probability": ActivationObserver(
                config.activation_bits, config.activation_percentile, config.histogram_bins, signed=False
            ),
            "value": ActivationObserver(config.activation_bits, config.activation_percentile, config.histogram_bins),
        }
        self.ranges: dict[str, float] = {}

    def apply(self, name: str, x: torch.Tensor) -> torch.Tensor:
        if self.state == UNCALIBRATED:
            raise RuntimeError(
                "PTQ attention was used before calibration. Call "
                "calibrate_ptq_model(...) and then finalize_ptq_model(...)."
            )
        if self.state == OBSERVING:
            self.observers[name].observe(x)
            return x
        return fake_quantize_activation(
            x,
            self.ranges[name],
            self.config.activation_bits,
            signed=self.observers[name].signed,
        )

    @property
    def _unused(self) -> set[str]:
        """Observers this configuration never feeds.

        Log-Int-Softmax puts the attention probabilities on a power-of-two grid
        determined by the softmax's own (0, 1) bound, so it observes no range.
        That observer is then legitimately empty, and finalizing must skip it
        rather than treat it as an uncalibrated site -- while still refusing to
        skip one that was merely never reached.
        """

        return {"probability"} if self.config.softmax_log2_bits is not None else set()

    def finalize(self) -> dict[str, float]:
        if self.state != QUANTIZED:
            self.ranges = {
                name: obs.resolve()
                for name, obs in self.observers.items()
                if name not in self._unused
            }
            self.state = QUANTIZED
        return self.ranges


def replace_ptq_attention_matmuls(
    module: nn.Module,
    config: TorchPTQConfig,
    mode: str = "ptq-full",
) -> int:
    """Install a W8A8 attention path, mirroring ``replace_pao_attention_matmuls``.

    There is no ``approximate_softmax`` counterpart: PTQ leaves the softmax in
    floating point by construction, which is the whole reason this backend is
    restricted to the ``multiplication`` scope.
    """

    valid_modes = {"exact", "ptq-qk", "ptq-pv", "ptq-full"}
    if mode not in valid_modes:
        raise ValueError(f"Unknown attention mode {mode!r}. Choose one of {sorted(valid_modes)}.")

    attention_modules = [
        child for child in module.modules() if isinstance(child, (ViTAttentionModule, GPT2Attention))
    ]
    if not attention_modules:
        return 0

    interface_name = f"ptq_{id(config)}_{mode}_{id(module)}"

    def ptq_attention_forward(
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

        observers: PTQAttentionObservers = attention_module._ptq_observers
        use_qk = mode in {"ptq-qk", "ptq-full"}
        use_pv = mode in {"ptq-pv", "ptq-full"}

        if use_qk:
            query_q = observers.apply("query", query)
            key_q = observers.apply("key", key)
            attention_scores = torch.matmul(query_q, key_q.transpose(-1, -2))
        else:
            attention_scores = torch.matmul(query, key.transpose(-1, -2))
        # Left exact on purpose: an int8 attention kernel folds this constant
        # into the dequantization of the int32 accumulator, so it costs no
        # multiplier. See the module docstring; the PAO and B-PLA paths do route
        # it through their approximate multipliers.
        attention_scores = attention_scores * scaling

        attention_weights = attention_scores
        if attention_mask is not None:
            attention_weights = attention_weights + attention_mask
        attention_weights = F.softmax(attention_weights, dim=-1)
        attention_weights = attention_weights.type(value.dtype)
        attention_weights = F.dropout(
            attention_weights, p=dropout, training=attention_module.training
        )
        if use_pv:
            if config.softmax_log2_bits is not None:
                # FQ-ViT's Log-Int-Softmax. Bounded to (0, 1), so it needs no
                # observed range and stays correct even in the observing state.
                from modules.torch_fqvit import log2_quantize_attention

                probabilities_q = log2_quantize_attention(
                    attention_weights, config.softmax_log2_bits
                )
            else:
                probabilities_q = observers.apply("probability", attention_weights)
            value_q = observers.apply("value", value)
            attention_output = torch.matmul(probabilities_q, value_q)
        else:
            attention_output = torch.matmul(attention_weights, value)
        return attention_output.transpose(1, 2), attention_weights

    ALL_ATTENTION_FUNCTIONS.register(interface_name, ptq_attention_forward)
    if AttentionMaskInterface is not None and eager_mask is not None:
        AttentionMaskInterface.register(interface_name, eager_mask)
    for attention_module in attention_modules:
        attention_module._ptq_observers = PTQAttentionObservers(config)
        attention_module.config._attn_implementation = interface_name
    module._ptq_attention_mode = mode
    return len(attention_modules)


# --------------------------------------------------------------------- replacers


def replace_ptq_linear(
    module: nn.Module,
    config: TorchPTQConfig,
    max_linear_modules: int | None = None,
    replace_conv2d: bool = False,
) -> int:
    """In-place W8A8 replacement mirroring ``replace_pao_linear_and_gelu``.

    No ``replace_gelu`` counterpart exists: W8A8 PTQ quantizes weights and
    activations and leaves the nonlinearities in floating point. Quantizing them
    too is a different research line (FQ-ViT, I-ViT) with its own baselines.
    """

    replaced = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear) and (
            max_linear_modules is None or replaced < max_linear_modules
        ):
            setattr(module, name, TorchPTQLinear(child, config))
            replaced += 1
            continue
        if replace_conv2d and isinstance(child, nn.Conv2d):
            setattr(module, name, TorchPTQConv2d(child, config))
            replaced += 1
            continue
        replaced += replace_ptq_linear(
            child,
            config=config,
            max_linear_modules=None if max_linear_modules is None else max_linear_modules - replaced,
            replace_conv2d=replace_conv2d,
        )
    return replaced


def replace_ptq_gpt2_conv1d(
    module: nn.Module,
    config: TorchPTQConfig,
    max_conv1d_modules: int | None = None,
    replace_lm_head: bool = False,
) -> int:
    """In-place W8A8 replacement mirroring ``replace_pao_gpt2_conv1d_and_gelu``."""

    replaced = 0
    for name, child in list(module.named_children()):
        if isinstance(child, Conv1D) and (
            max_conv1d_modules is None or replaced < max_conv1d_modules
        ):
            setattr(module, name, TorchPTQConv1D(child, config))
            replaced += 1
            continue
        if replace_lm_head and isinstance(child, nn.Linear):
            setattr(module, name, TorchPTQLinear(child, config))
            replaced += 1
            continue
        replaced += replace_ptq_gpt2_conv1d(
            child,
            config=config,
            max_conv1d_modules=None if max_conv1d_modules is None else max_conv1d_modules - replaced,
            replace_lm_head=replace_lm_head,
        )
    return replaced


# ------------------------------------------------------------------ calibration


def _ptq_sites(model: nn.Module) -> tuple[list[nn.Module], list[PTQAttentionObservers]]:
    """Collect every calibratable site, by protocol rather than by class.

    The FQ-ViT LayerNorm proxy lives in a module that imports this one, so it
    cannot be named here without a cycle. It carries the same three-state
    ``ptq_state`` and the same ``_finalize`` and ``observer`` members, which is
    all the calibration driver needs.
    """

    layers = [
        child
        for child in model.modules()
        if hasattr(child, "ptq_state") and hasattr(child, "_finalize")
    ]
    attentions = [
        child._ptq_observers
        for child in model.modules()
        if isinstance(getattr(child, "_ptq_observers", None), PTQAttentionObservers)
    ]
    return layers, attentions


def set_ptq_state(model: nn.Module, state: str) -> int:
    """Move every PTQ site in ``model`` into ``state``; returns the site count."""

    if state not in {UNCALIBRATED, OBSERVING, QUANTIZED}:
        raise ValueError(f"Unknown PTQ state {state!r}.")
    layers, attentions = _ptq_sites(model)
    for layer in layers:
        layer.ptq_state = state
    for attention in attentions:
        attention.state = state
    return len(layers) + len(attentions)


@torch.no_grad()
def calibrate_ptq_model(
    model: nn.Module,
    batches: Iterable[Any],
    forward_batch: Callable[[nn.Module, Any], Any],
    max_batches: int,
) -> int:
    """Observe activation ranges over ``max_batches`` forward passes.

    Signature deliberately matches ``calibrate_model_activation_range`` so the
    harness can hand both backends the same batches, the same count, and the
    same forward closure. Labels are never touched and no weight is updated;
    during calibration the model computes exactly what the float model computes,
    so the observed ranges are the float model's, not a partially quantized
    model's.
    """

    sites = set_ptq_state(model, OBSERVING)
    if sites == 0:
        raise RuntimeError("calibrate_ptq_model() found no PTQ modules to calibrate.")
    model.eval()
    seen = 0
    for index, batch in enumerate(batches):
        if index >= max_batches:
            break
        forward_batch(model, batch)
        seen += 1
    if seen == 0:
        raise RuntimeError("calibrate_ptq_model() ran zero batches; nothing was observed.")
    return seen


def finalize_ptq_model(model: nn.Module) -> dict[str, object]:
    """Resolve every observed range into a scale and switch the model to W8A8.

    Raises if any site never saw an activation, rather than silently leaving it
    in a pass-through state.
    """

    layers, attentions = _ptq_sites(model)
    if not layers and not attentions:
        raise RuntimeError("finalize_ptq_model() found no PTQ modules.")
    uncalibrated = [
        type(layer).__name__ for layer in layers if not layer.observer.calibrated
    ]
    if uncalibrated:
        raise RuntimeError(
            f"{len(uncalibrated)} PTQ module(s) saw no activation during calibration "
            f"(e.g. {uncalibrated[0]}). The calibration batches do not exercise the "
            "whole model."
        )
    layer_ranges = [layer._finalize() for layer in layers]
    attention_ranges = [attention.finalize() for attention in attentions]
    return {
        "quantized_layers": len(layers),
        "quantized_attention_blocks": len(attentions),
        "activation_range_min": min(layer_ranges) if layer_ranges else None,
        "activation_range_max": max(layer_ranges) if layer_ranges else None,
        "attention_probability_range_max": (
            max(r["probability"] for r in attention_ranges)
            if attention_ranges and all("probability" in r for r in attention_ranges)
            else None
        ),
    }
