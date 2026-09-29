"""
SmoothQuant activation-outlier migration, as a front-end to the W8A8 backend.

Reference implementation of the smoothing step defined in Xiao et al.,
"SmoothQuant: Accurate and Efficient Post-Training Quantization for Large
Language Models", ICML 2023 (official code: https://github.com/mit-han-lab/
smoothquant, ``smoothquant/smooth.py``).

Why this module exists
----------------------
Our ``ptq-w8a8`` backend already uses dynamic per-token activation scales, which
is the granularity SmoothQuant's O1 setting uses. What it does *not* do is the
part that gives SmoothQuant its name: migrating activation outliers into the
weights before quantizing. Adding that migration in front of the existing
backend turns it into SmoothQuant while every other backend in a matched run
keeps sharing one float32 accumulator, one chunking rule, one sample order and
one reference. Swapping in the official ``W8A8Linear`` instead would change the
arithmetic as well as the method, and the table could no longer attribute a
difference to the method alone.

What the smoothing does
-----------------------
For a LayerNorm whose output feeds a group of linear projections, and a
per-input-channel activation maximum ``a_j`` collected on calibration data,

    s_j = a_j^alpha / w_j^(1 - alpha),      w_j = max_over_outputs |W[:, j]|

then ``LN.weight /= s``, ``LN.bias /= s``, and ``W[:, j] *= s_j``. In exact
arithmetic this is the identity -- the model's outputs do not change at all --
but the activations entering the quantizer are flattened and the difficulty is
moved into the weights, which are far easier to quantize per channel.

The identity only holds when the LayerNorm's output feeds nothing but the group.
That is true for every pair used here: GPT-2's ``ln_1 -> c_attn`` and
``ln_2 -> c_fc``, and ViT's ``layernorm_before -> {query, key, value}`` and
``layernorm_after -> intermediate.dense``. Both architectures take their
residual from the pre-norm tensor, not the normalized one.

Weight layouts differ and the fold has to follow them: ``nn.Linear`` stores
``[out, in]`` so an input channel is a column, while HuggingFace ``Conv1D``
stores ``[in, out]`` so an input channel is a row. Getting this backwards is
silent -- the shapes still broadcast -- so the two cases are separated below and
``tests/test_smoothquant.py`` pins the identity numerically for each.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Sequence

import torch
import torch.nn as nn

try:  # pragma: no cover - only absent if transformers is not installed
    from transformers.pytorch_utils import Conv1D
except Exception:  # pragma: no cover
    Conv1D = ()  # type: ignore[assignment]


@dataclass
class SmoothQuantConfig:
    #: Migration strength. 0 leaves activations untouched, 1 moves the whole
    #: outlier into the weights. Xiao et al. use 0.5 for OPT and recommend it as
    #: the default; 0.85 is their setting for the harder LLaMA-family models.
    alpha: float = 0.5
    #: Scales below this are clamped. A channel that is identically zero on the
    #: calibration set would otherwise divide the LayerNorm by zero.
    min_scale: float = 1e-5


class ChannelMaxObserver:
    """Per-input-channel ``max |x|``, accumulated over calibration batches.

    SmoothQuant needs the channel profile, not a single range, which is what
    separates it from the per-tensor observer in ``torch_ptq``. Only the maximum
    is kept, so memory is one vector per group regardless of how much
    calibration data is fed through.
    """

    def __init__(self) -> None:
        self.maximum: torch.Tensor | None = None

    @torch.no_grad()
    def observe(self, x: torch.Tensor) -> None:
        flat = x.detach().reshape(-1, x.shape[-1]).abs().amax(dim=0).float()
        if self.maximum is None:
            self.maximum = flat.clone()
        else:
            self.maximum = torch.maximum(self.maximum, flat)


def _weight_input_maxima(module: nn.Module) -> torch.Tensor:
    """``max |W[:, j]|`` over output channels, for input channel ``j``."""

    weight = module.weight.detach()
    if isinstance(module, nn.Linear):
        return weight.abs().amax(dim=0).float()          # [out, in] -> per column
    if Conv1D and isinstance(module, Conv1D):
        return weight.abs().amax(dim=1).float()          # [in, out] -> per row
    raise TypeError(f"Cannot read input-channel maxima from {type(module).__name__}.")


def _scale_weight_inputs(module: nn.Module, scale: torch.Tensor) -> None:
    """Multiply input channel ``j`` of ``module`` by ``scale[j]``, in place."""

    weight = module.weight
    if isinstance(module, nn.Linear):
        weight.data.mul_(scale.to(weight.dtype).unsqueeze(0))     # broadcast over rows
    elif Conv1D and isinstance(module, Conv1D):
        weight.data.mul_(scale.to(weight.dtype).unsqueeze(1))     # broadcast over columns
    else:
        raise TypeError(f"Cannot scale input channels of {type(module).__name__}.")


@torch.no_grad()
def smooth_group(
    norm: nn.LayerNorm,
    projections: Sequence[nn.Module],
    activation_max: torch.Tensor,
    config: SmoothQuantConfig,
) -> torch.Tensor:
    """Fold one LayerNorm/projection group and return the scales applied."""

    device, dtype = norm.weight.device, norm.weight.dtype
    activation_max = activation_max.to(device=device, dtype=torch.float32)

    weight_max = torch.stack(
        [_weight_input_maxima(p).to(device) for p in projections]
    ).amax(dim=0)

    if activation_max.shape != weight_max.shape:
        raise ValueError(
            f"Activation profile {tuple(activation_max.shape)} does not match the "
            f"projection input width {tuple(weight_max.shape)}."
        )

    alpha = config.alpha
    scale = (
        activation_max.clamp(min=config.min_scale).pow(alpha)
        / weight_max.clamp(min=config.min_scale).pow(1.0 - alpha)
    ).clamp(min=config.min_scale)

    norm.weight.data.div_(scale.to(dtype))
    if norm.bias is not None:
        norm.bias.data.div_(scale.to(dtype))
    for projection in projections:
        _scale_weight_inputs(projection, scale)
    return scale


def _projection_input_width(module: nn.Module) -> int | None:
    if isinstance(module, nn.Linear):
        return module.in_features
    if Conv1D and isinstance(module, Conv1D):
        return int(module.weight.shape[0])          # Conv1D stores [in, out]
    return None


def _modules_inside_repeated_blocks(model: nn.Module) -> set[int]:
    """Ids of every module living under some ``nn.ModuleList`` entry.

    SmoothQuant folds the LayerNorm/projection pairs *inside* transformer
    blocks. The model-final norm -- GPT-2's ``ln_f``, ViT's ``layernorm`` --
    feeds the classifier and is left alone, which is what the official
    implementation does. Repeated blocks live in a ``ModuleList`` in every
    architecture we target, so that containment is the test, rather than a name
    that changes between library versions.
    """

    inside: set[int] = set()
    for container in model.modules():
        if isinstance(container, nn.ModuleList):
            for block in container:
                for descendant in block.modules():
                    inside.add(id(descendant))
    return inside


@torch.no_grad()
def discover_groups(
    model: nn.Module,
    sample_batch: Any,
    forward_batch: Callable[[nn.Module, Any], Any],
) -> list[tuple[nn.LayerNorm, list[nn.Module]]]:
    """Find LayerNorm/projection groups by watching one forward pass.

    Earlier versions of this module walked fixed attribute paths --
    ``block.ln_1.attn.c_attn``, ``layer.layernorm_before`` and so on -- and broke
    the first time the library renamed a field. Matching a LayerNorm's output
    tensor against the inputs the projections actually receive finds the same
    pairs from the dataflow itself, so it survives renames and covers both
    architectures with one path.

    The fold is an identity only if nothing outside the group reads the
    normalized tensor. That holds for GPT-2 and ViT, which take their residual
    from the pre-norm tensor; an architecture that fed a residual from the
    LayerNorm output would need this checked before smoothing.
    """

    # Events in execution order. A bare {pointer: module} map would be wrong:
    # the caching allocator reuses an address once the tensor holding it is
    # freed, so a later layer's LayerNorm can land on an earlier one's address.
    # Freeing only happens after that tensor's consumers have run, so pairing
    # each projection with the *most recent preceding* LayerNorm of the same
    # address is unambiguous even when addresses repeat.
    events: list[tuple[str, nn.Module, int]] = []
    handles = []

    def norm_hook(module: nn.Module, _inputs: Any, output: Any) -> None:
        if isinstance(output, torch.Tensor):
            events.append(("norm", module, output.data_ptr()))

    def projection_hook(module: nn.Module, inputs: Any) -> None:
        if inputs and isinstance(inputs[0], torch.Tensor):
            events.append(("projection", module, inputs[0].data_ptr()))

    inside_blocks = _modules_inside_repeated_blocks(model)
    for module in model.modules():
        if isinstance(module, nn.LayerNorm):
            handles.append(module.register_forward_hook(norm_hook))
        elif _projection_input_width(module) is not None:
            handles.append(module.register_forward_pre_hook(projection_hook))

    try:
        model.eval()
        forward_batch(model, sample_batch)
    finally:
        for handle in handles:
            handle.remove()

    collected: dict[int, tuple[nn.LayerNorm, list[nn.Module]]] = {}
    latest: dict[int, nn.LayerNorm] = {}
    norm_count = 0
    projection_count = 0
    for kind, module, pointer in events:
        if kind == "norm":
            norm_count += 1
            latest[pointer] = module                    # type: ignore[assignment]
            continue
        projection_count += 1
        norm = latest.get(pointer)
        if norm is None or id(norm) not in inside_blocks:
            continue
        if _projection_input_width(module) != int(norm.normalized_shape[-1]):
            continue
        entry = collected.setdefault(id(norm), (norm, []))
        if module not in entry[1]:
            entry[1].append(module)

    groups = list(collected.values())
    if not groups:
        raise ValueError(
            "SmoothQuant found no LayerNorm/projection groups. Observed "
            f"{norm_count} LayerNorm outputs, {projection_count} projection inputs, "
            f"{len(inside_blocks)} modules inside repeated blocks. Either the "
            "checkpoint has no pre-norm transformer blocks, or a LayerNorm output "
            "is copied before it reaches its projections."
        )
    return groups


def collect_group_activation_maxima(
    model: nn.Module,
    groups: Sequence[tuple[nn.LayerNorm, list[nn.Module]]],
    batches: Iterable[Any],
    forward_batch: Callable[[nn.Module, Any], Any],
    max_batches: int,
) -> list[torch.Tensor]:
    """Per-channel activation maxima entering each group, on calibration data.

    The hook sits on the LayerNorm's *output*, which is exactly the tensor the
    projections consume, so one hook covers a whole group. Calibration is
    forward-only and sees no labels, matching what every other backend gets.
    """

    observers = [ChannelMaxObserver() for _ in groups]
    handles = []

    def make_hook(observer: ChannelMaxObserver):
        def hook(_module: nn.Module, _inputs: Any, output: torch.Tensor) -> None:
            observer.observe(output)

        return hook

    for (norm, _), observer in zip(groups, observers):
        handles.append(norm.register_forward_hook(make_hook(observer)))

    try:
        model.eval()
        with torch.no_grad():
            for index, batch in enumerate(batches):
                if index >= max_batches:
                    break
                forward_batch(model, batch)
    finally:
        for handle in handles:
            handle.remove()

    missing = [i for i, o in enumerate(observers) if o.maximum is None]
    if missing:
        raise RuntimeError(
            f"{len(missing)} LayerNorm groups saw no calibration activations; "
            "the group wiring does not match this checkpoint."
        )
    return [observer.maximum for observer in observers]  # type: ignore[misc]


def smooth_model(
    model: nn.Module,
    batches: Iterable[Any],
    forward_batch: Callable[[nn.Module, Any], Any],
    max_batches: int,
    config: SmoothQuantConfig,
    is_gpt2: bool = False,
) -> dict[str, object]:
    """Apply SmoothQuant migration in place; report what was touched.

    Runs before any quantization: the model is still exact here, so the
    activation profile is the one the unquantized checkpoint actually produces.
    ``is_gpt2`` is accepted for call-site symmetry with the other backends and
    is no longer needed -- the groups come from the dataflow, not the family.
    """

    batches = list(batches)
    if not batches:
        raise ValueError("SmoothQuant needs at least one calibration batch.")
    groups = discover_groups(model, batches[0], forward_batch)
    maxima = collect_group_activation_maxima(
        model, groups, batches, forward_batch, max_batches
    )
    scales = [
        smooth_group(norm, projections, activation_max, config)
        for (norm, projections), activation_max in zip(groups, maxima)
    ]
    stacked = torch.cat([s.reshape(-1) for s in scales])
    return {
        "smoothquant_groups": len(groups),
        "smoothquant_projections": sum(len(p) for _, p in groups),
        "smoothquant_alpha": config.alpha,
        "smoothquant_scale_min": float(stacked.min()),
        "smoothquant_scale_max": float(stacked.max()),
        "smoothquant_scale_mean": float(stacked.mean()),
    }
