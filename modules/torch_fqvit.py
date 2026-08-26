"""
FQ-ViT (Lin et al., IJCAI 2022) baseline: Power-of-Two Factor and Log-Int-Softmax.

FQ-ViT matters to this project because it is the only training-free method that
reaches the nonlinear operators at all. Standard W8A8 quantizes weights and
activations and leaves GELU, Softmax and LayerNorm in floating point; FQ-ViT
quantizes LayerNorm and Softmax too, with calibration only and no gradient step,
which puts it in the same setting as B-PLA. The methods that go further --
I-BERT and I-ViT -- both require quantization-aware fine-tuning, so they are not
comparable here.

Coverage, and why it has to be stated
------------------------------------
FQ-ViT does not address GELU. The paper never mentions it. Its coverage
therefore sits *between* this harness's ``multiplication`` scope and B-PLA's
``combined`` scope, and any table putting the two side by side has to say so --
otherwise a scope difference reads as a fidelity difference.

Faithfulness
------------
Both mechanisms are reimplemented from the released code
(github.com/megvii-research/FQ-ViT), not from the paper's prose, because the
paper leaves the per-channel scale search underspecified.

* ``PowerOfTwoFactorObserver`` follows ``PtfObserver.get_quantization_params``:
  one layer-wise base scale, then a per-channel power-of-two divisor chosen by
  squared error over the calibration activations.
* ``log2_quantize_attention`` follows ``Log2Quantizer``: ``round(-log2 p)``,
  clamped to the bit width, with everything past the range flushed to zero.

Deliberate deviation, in FQ-ViT's favour
----------------------------------------
FQ-ViT's ``QIntLayerNorm`` recomputes the normalization itself in integer
arithmetic and quantizes the resulting affine factor to an 8-bit dyadic number.
That is a further approximation stacked on top of quantizing the input. We
reproduce the input quantization -- which is what PTF exists for and what the
paper argues about -- and leave the normalization in float. These numbers are
therefore an upper bound on what FQ-ViT achieves, which is the direction a
baseline we implemented ourselves should err in.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.torch_ptq import (
    OBSERVING,
    QUANTIZED,
    UNCALIBRATED,
    _TINY,
    TorchPTQConfig,
    _validate_config,
)


class PowerOfTwoFactorObserver:
    """Per-channel power-of-two sub-scales for a LayerNorm input.

    FQ-ViT's observation is that LayerNorm inputs vary far more across channels
    than within them, which is what breaks a single layer-wise scale. Rather
    than spend a full independent scale per channel -- which an integer pipeline
    cannot factor back out of the normalization -- PTF gives every channel the
    *same* base scale divided by a power of two, so the per-channel correction
    costs a shift and the layer keeps one scale.

    The exponent is chosen per channel by squared error against the unquantized
    values. We keep a bounded sample of the calibration activations rather than
    all of them: the search is over four candidates per channel and does not
    need every token to resolve.
    """

    #: Rows of activations retained per site for the scale search.
    SAMPLE_ROWS = 4096

    def __init__(self, bits: int = 8, factors: int = 4):
        self.bits = bits
        self.factors = factors
        self.min_val: torch.Tensor | None = None
        self.max_val: torch.Tensor | None = None
        self._sample: torch.Tensor | None = None

    @property
    def calibrated(self) -> bool:
        return self._sample is not None

    @torch.no_grad()
    def observe(self, x: torch.Tensor) -> None:
        flat = x.detach().reshape(-1, x.shape[-1]).float()
        if flat.numel() == 0:
            return
        batch_min = flat.min(dim=0).values
        batch_max = flat.max(dim=0).values
        self.min_val = batch_min if self.min_val is None else torch.minimum(self.min_val, batch_min)
        self.max_val = batch_max if self.max_val is None else torch.maximum(self.max_val, batch_max)
        if self._sample is None:
            self._sample = flat[: self.SAMPLE_ROWS].clone()
        elif self._sample.shape[0] < self.SAMPLE_ROWS:
            room = self.SAMPLE_ROWS - self._sample.shape[0]
            if room > 0:
                self._sample = torch.cat([self._sample, flat[:room]], dim=0)

    @torch.no_grad()
    def resolve(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(per-channel scale, zero point)``.

        Asymmetric and unsigned, matching FQ-ViT: a LayerNorm input is not
        centred, so a symmetric grid would leave most of its range unused.
        """

        if self._sample is None or self.max_val is None or self.min_val is None:
            raise RuntimeError(
                "PowerOfTwoFactorObserver.resolve() was called before any activation "
                "was observed. Run the calibration pass first."
            )
        qmin, qmax = 0.0, float(2**self.bits - 1)
        base = ((self.max_val.max() - self.min_val.min()) / (qmax - qmin)).clamp(min=_TINY)
        zero_point = (qmin - torch.round(self.min_val.min() / base)).clamp(qmin, qmax)

        sample = self._sample
        errors = []
        for exponent in range(self.factors):
            scale = base / (2.0**exponent)
            dequantized = (
                torch.clamp(torch.round(sample / scale + zero_point), qmin, qmax) - zero_point
            ) * scale
            errors.append((dequantized - sample).pow(2).sum(dim=0))
        best = torch.stack(errors, dim=0).argmin(dim=0)
        return base / torch.pow(2.0, best.to(sample.dtype)), zero_point


def fake_quantize_power_of_two_factor(
    x: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    bits: int = 8,
) -> torch.Tensor:
    """Asymmetric per-channel quantize-dequantize along the last dimension."""

    qmin, qmax = 0.0, float(2**bits - 1)
    scale = scale.to(device=x.device, dtype=x.dtype).clamp(min=_TINY)
    zero_point = zero_point.to(device=x.device, dtype=x.dtype)
    quantized = torch.clamp(torch.round(x / scale + zero_point), qmin, qmax)
    return (quantized - zero_point) * scale


def log2_quantize_attention(probabilities: torch.Tensor, bits: int = 4) -> torch.Tensor:
    """Log-Int-Softmax: put attention probabilities on a power-of-two grid.

    ``round(-log2 p)`` is an integer exponent, so the PV matmul becomes a shift
    instead of a multiply, and the grid is finest exactly where softmax puts its
    mass. Probabilities below ``2^-(2^bits)`` fall past the representable range
    and are flushed to zero, which is what the released implementation does
    through its ``softmax_mask``.

    Note this is calibration-free: softmax output is bounded to (0, 1), so no
    range has to be observed. That is one of the two properties the paper argues
    makes log2 quantization the right fit for attention.
    """

    levels = 2**bits
    exponent = torch.round(-torch.log2(probabilities.clamp(min=_TINY)))
    underflow = exponent >= levels
    dequantized = torch.pow(2.0, -torch.clamp(exponent, 0, levels - 1))
    return torch.where(underflow, torch.zeros_like(dequantized), dequantized)


class TorchFQViTLayerNorm(nn.Module):
    """LayerNorm whose input is quantized with FQ-ViT's Power-of-Two Factor.

    See the module docstring: the normalization arithmetic stays in float, so
    this is an upper bound on FQ-ViT rather than a reproduction of its integer
    path.
    """

    _span_coverage_kinds = frozenset({"multiply", "transcendental", "normalization"})

    def __init__(self, source: nn.LayerNorm, config: TorchPTQConfig):
        super().__init__()
        _validate_config(config)
        self.config = config
        self.normalized_shape = tuple(source.normalized_shape)
        self.eps = float(source.eps)
        self.ptq_state = UNCALIBRATED
        self.observer = PowerOfTwoFactorObserver(bits=config.activation_bits)
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
        self.register_buffer("input_scale", torch.zeros(int(self.normalized_shape[-1])))
        self.register_buffer("input_zero_point", torch.zeros(()))

    def _finalize(self) -> float:
        if self.ptq_state == QUANTIZED:
            return float(self.input_scale.max())
        scale, zero_point = self.observer.resolve()
        self.input_scale = scale.to(device=self.input_scale.device, dtype=torch.float32)
        self.input_zero_point = zero_point.to(
            device=self.input_zero_point.device, dtype=torch.float32
        )
        self.ptq_state = QUANTIZED
        return float(self.input_scale.max())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.ptq_state == UNCALIBRATED:
            raise RuntimeError(
                "TorchFQViTLayerNorm was used before calibration. Call "
                "calibrate_ptq_model(...) and then finalize_ptq_model(...)."
            )
        if self.ptq_state == OBSERVING:
            self.observer.observe(x)
        else:
            x = fake_quantize_power_of_two_factor(
                x, self.input_scale, self.input_zero_point, self.config.activation_bits
            )
        return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)


def replace_fqvit_layer_norms(module: nn.Module, config: TorchPTQConfig) -> int:
    """Replace every ``nn.LayerNorm`` with the Power-of-Two Factor proxy."""

    replaced = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.LayerNorm):
            setattr(module, name, TorchFQViTLayerNorm(child, config))
            replaced += 1
        else:
            replaced += replace_fqvit_layer_norms(child, config)
    module._fqvit_layernorm_count = replaced
    return replaced
