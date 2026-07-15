"""Compute-only theoretical energy accounting for B-PLA experiments.

The model deliberately excludes coefficient tables, registers, SRAM, DRAM,
interconnect, leakage, and software runtime.  Default arithmetic energies are
the often-used 45 nm illustrative values from Horowitz (ISSCC 2014,
doi:10.1109/ISSCC.2014.6757323). Results are sensitivity estimates, not
measurements or synthesis claims.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ComputeEnergyTablePJ:
    """Energy per arithmetic/control action in picojoules."""

    fp32_mul: float = 3.70
    fp32_add: float = 0.90
    int32_add: float = 0.10
    fixed_shift: float = 0.0
    small_control: float = 0.005
    fp32_tanh: float = 0.0
    # Optimistic configurable proxies. Horowitz (2014) does not provide the
    # exact exp/reciprocal/rsqrt units modeled by these PyTorch operators, so
    # each defaults to one FP32-multiply equivalent rather than inventing a
    # hardware-specific transcendental-unit claim.
    fp32_exp: float = 3.70
    fp32_reciprocal: float = 3.70
    fp32_rsqrt: float = 3.70

    def fixed_add(self, bits: int) -> float:
        return self.int32_add * bits / 32.0


@dataclass(frozen=True)
class BPLAComputeConfig:
    affine_path: str = "dyadic"
    dyadic_terms: int = 2
    mantissa_bits: int = 24

    def __post_init__(self) -> None:
        if self.affine_path not in {"float", "dyadic"}:
            raise ValueError("affine_path must be 'float' or 'dyadic'.")
        if self.dyadic_terms <= 0:
            raise ValueError("dyadic_terms must be positive.")
        if self.mantissa_bits <= 0:
            raise ValueError("mantissa_bits must be positive.")


@dataclass(frozen=True)
class ComputeWorkload:
    """Modeled arithmetic sites, including row-wise normalization operators."""

    multiply_sites: int
    bpla_multiply_sites: int
    gelu_sites: int
    bpla_gelu_sites: int
    label: str
    softmax_rows: int = 0
    softmax_elements: int = 0
    bpla_softmax_rows: int = 0
    bpla_softmax_elements: int = 0
    layernorm_rows: int = 0
    layernorm_elements: int = 0
    bpla_layernorm_rows: int = 0
    bpla_layernorm_elements: int = 0

    def __post_init__(self) -> None:
        values = (
            self.multiply_sites,
            self.bpla_multiply_sites,
            self.gelu_sites,
            self.bpla_gelu_sites,
            self.softmax_rows,
            self.softmax_elements,
            self.bpla_softmax_rows,
            self.bpla_softmax_elements,
            self.layernorm_rows,
            self.layernorm_elements,
            self.bpla_layernorm_rows,
            self.bpla_layernorm_elements,
        )
        if any(value < 0 for value in values):
            raise ValueError("workload counts must be non-negative.")
        if self.bpla_multiply_sites > self.multiply_sites:
            raise ValueError("B-PLA multiplication sites cannot exceed total sites.")
        if self.bpla_gelu_sites > self.gelu_sites:
            raise ValueError("B-PLA GELU sites cannot exceed total sites.")
        if self.bpla_softmax_rows > self.softmax_rows or self.bpla_softmax_elements > self.softmax_elements:
            raise ValueError("B-PLA Softmax sites cannot exceed total sites.")
        if self.bpla_layernorm_rows > self.layernorm_rows or self.bpla_layernorm_elements > self.layernorm_elements:
            raise ValueError("B-PLA LayerNorm sites cannot exceed total sites.")
        if (self.softmax_rows == 0) != (self.softmax_elements == 0):
            raise ValueError("Softmax rows and elements must both be zero or positive.")
        if (self.layernorm_rows == 0) != (self.layernorm_elements == 0):
            raise ValueError("LayerNorm rows and elements must both be zero or positive.")


def fp32_gelu_energy_pj(table: ComputeEnergyTablePJ) -> dict[str, float]:
    """Lower-bound cost of the tanh-form GELU used by this project.

    0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3))) contains six FP
    multiplications and two FP additions when constants are implemented as
    multiplications.  ``fp32_tanh`` is exposed separately and defaults to zero,
    making the baseline deliberately favorable to conventional GELU.
    """

    mul_count = 6
    add_count = 2
    total = mul_count * table.fp32_mul + add_count * table.fp32_add + table.fp32_tanh
    return {
        "fp32_mul_count": float(mul_count),
        "fp32_add_count": float(add_count),
        "tanh_energy_pj": table.fp32_tanh,
        "total_pj": total,
    }


def bpla_multiplier_energy_pj(
    config: BPLAComputeConfig,
    table: ComputeEnergyTablePJ,
) -> dict[str, float]:
    """Energy for product generation only; dot-product accumulation is separate."""

    if config.affine_path == "float":
        # a*m1 + b*m2 + c, then 1 + m1 + m2 + cross.
        fp_mul_count = 2
        fp_add_count = 5
        fixed_shift_count = 0
        fixed_add_count = 0
        arithmetic = fp_mul_count * table.fp32_mul + fp_add_count * table.fp32_add
    else:
        # Current software represents a, b, and c with T signed-POT terms.
        # Reducing 3T terms takes 3T-1 adds; mantissa reconstruction takes 3.
        fp_mul_count = 0
        fp_add_count = 0
        fixed_shift_count = 3 * config.dyadic_terms
        fixed_add_count = 3 * config.dyadic_terms + 2
        arithmetic = (
            fixed_shift_count * table.fixed_shift
            + fixed_add_count * table.fixed_add(config.mantissa_bits)
        )

    # Two 8-bit exponent additions expressed as fractions of one int32 add.
    exponent_energy = 0.5 * table.int32_add
    control_energy = table.small_control
    total = arithmetic + exponent_energy + control_energy
    return {
        "affine_path": config.affine_path,
        "dyadic_terms": float(config.dyadic_terms),
        "fp32_mul_count": float(fp_mul_count),
        "fp32_add_count": float(fp_add_count),
        "fixed_shift_count": float(fixed_shift_count),
        "fixed_add_count": float(fixed_add_count),
        "arithmetic_energy_pj": arithmetic,
        "exponent_energy_pj": exponent_energy,
        "control_energy_pj": control_energy,
        "total_pj": total,
        "fp32_mul_pj": table.fp32_mul,
        "ratio_to_fp32_mul": total / table.fp32_mul,
        "savings_vs_fp32_mul_pct": 100.0 * (1.0 - total / table.fp32_mul),
    }


def bpla_gelu_energy_pj(
    config: BPLAComputeConfig,
    table: ComputeEnergyTablePJ,
) -> dict[str, float]:
    """Compute-only energy for one B-PLA GELU affine evaluation."""

    if config.affine_path == "float":
        fp_mul_count = 1
        fp_add_count = 1
        fixed_shift_count = 0
        fixed_add_count = 0
        arithmetic = table.fp32_mul + table.fp32_add
    else:
        # Slope and intercept both contain T signed-POT terms.
        fp_mul_count = 0
        fp_add_count = 0
        fixed_shift_count = 2 * config.dyadic_terms
        fixed_add_count = 2 * config.dyadic_terms - 1
        arithmetic = (
            fixed_shift_count * table.fixed_shift
            + fixed_add_count * table.fixed_add(config.mantissa_bits)
        )

    control_energy = table.small_control
    total = arithmetic + control_energy
    baseline = fp32_gelu_energy_pj(table)["total_pj"]
    return {
        "affine_path": config.affine_path,
        "dyadic_terms": float(config.dyadic_terms),
        "fp32_mul_count": float(fp_mul_count),
        "fp32_add_count": float(fp_add_count),
        "fixed_shift_count": float(fixed_shift_count),
        "fixed_add_count": float(fixed_add_count),
        "arithmetic_energy_pj": arithmetic,
        "control_energy_pj": control_energy,
        "total_pj": total,
        "fp32_gelu_pj": baseline,
        "ratio_to_fp32_gelu": total / baseline,
        "savings_vs_fp32_gelu_pct": 100.0 * (1.0 - total / baseline),
    }


def fp32_softmax_energy_pj(
    elements: int,
    rows: int,
    table: ComputeEnergyTablePJ,
) -> dict[str, float]:
    """Optimistic conventional Softmax: exp, sum, reciprocal, and scaling."""

    if elements < rows or rows < 0:
        raise ValueError("Softmax elements must be at least its row count.")
    reduction_sites = elements - rows
    max_control = reduction_sites * table.small_control
    subtract = elements * table.fp32_add
    exponential = elements * table.fp32_exp
    denominator_sum = reduction_sites * table.fp32_add
    reciprocal = rows * table.fp32_reciprocal
    normalize = elements * table.fp32_mul
    total = max_control + subtract + exponential + denominator_sum + reciprocal + normalize
    return {
        "max_control_pj": max_control,
        "subtract_pj": subtract,
        "exp_pj": exponential,
        "sum_pj": denominator_sum,
        "reciprocal_pj": reciprocal,
        "normalize_pj": normalize,
        "total_pj": total,
    }


def bpla_softmax_energy_pj(
    elements: int,
    rows: int,
    config: BPLAComputeConfig,
    table: ComputeEnergyTablePJ,
) -> dict[str, float]:
    """Energy matching ``bpla_softmax_torch``, including its correction pass."""

    if elements < rows or rows < 0:
        raise ValueError("Softmax elements must be at least its row count.")
    reduction_sites = elements - rows
    nonlinear = bpla_gelu_energy_pj(config, table)["total_pj"]
    bpla_mul = bpla_multiplier_energy_pj(config, table)["total_pj"]
    common = (
        reduction_sites * table.small_control
        + elements * table.fp32_add
        + reduction_sites * table.fp32_add
    )
    base2_conversion = elements * table.fp32_mul
    exp2_fraction = elements * nonlinear
    exp2_shift = elements * table.fixed_shift
    first_reciprocal = rows * (nonlinear + table.fixed_shift)
    first_normalize = elements * bpla_mul
    correction_sum = reduction_sites * table.fp32_add
    correction_reciprocal = rows * (nonlinear + table.fixed_shift)
    correction_normalize = elements * bpla_mul
    total = (
        common
        + base2_conversion
        + exp2_fraction
        + exp2_shift
        + first_reciprocal
        + first_normalize
        + correction_sum
        + correction_reciprocal
        + correction_normalize
    )
    return {
        "common_pj": common,
        "base2_conversion_pj": base2_conversion,
        "exp2_fraction_pj": exp2_fraction,
        "exp2_shift_pj": exp2_shift,
        "first_reciprocal_pj": first_reciprocal,
        "first_normalize_pj": first_normalize,
        "correction_sum_pj": correction_sum,
        "correction_reciprocal_pj": correction_reciprocal,
        "correction_normalize_pj": correction_normalize,
        "total_pj": total,
    }


def fp32_layernorm_energy_pj(
    elements: int,
    rows: int,
    table: ComputeEnergyTablePJ,
) -> dict[str, float]:
    """LayerNorm arithmetic matching mean, variance, rsqrt, and affine output."""

    if elements < rows or rows < 0:
        raise ValueError("LayerNorm elements must be at least its row count.")
    add_sites = 4 * elements - rows
    mul_sites = 3 * elements + 2 * rows
    additions = add_sites * table.fp32_add
    multiplications = mul_sites * table.fp32_mul
    rsqrt = rows * table.fp32_rsqrt
    return {
        "fp32_add_sites": float(add_sites),
        "fp32_mul_sites": float(mul_sites),
        "additions_pj": additions,
        "multiplications_pj": multiplications,
        "rsqrt_pj": rsqrt,
        "total_pj": additions + multiplications + rsqrt,
    }


def bpla_layernorm_energy_pj(
    elements: int,
    rows: int,
    config: BPLAComputeConfig,
    table: ComputeEnergyTablePJ,
) -> dict[str, float]:
    """Energy matching the composed B-PLA LayerNorm implementation."""

    if elements < rows or rows < 0:
        raise ValueError("LayerNorm elements must be at least its row count.")
    add_sites = 4 * elements - rows
    multiply_sites = 3 * elements + 2 * rows
    additions = add_sites * table.fp32_add
    multiplication = multiply_sites * bpla_multiplier_energy_pj(config, table)["total_pj"]
    rsqrt = rows * (bpla_gelu_energy_pj(config, table)["total_pj"] + table.fixed_shift)
    return {
        "fp32_add_sites": float(add_sites),
        "bpla_multiply_sites": float(multiply_sites),
        "additions_pj": additions,
        "multiplications_pj": multiplication,
        "rsqrt_pj": rsqrt,
        "total_pj": additions + multiplication + rsqrt,
    }


def estimate_workload_compute_energy(
    workload: ComputeWorkload,
    config: BPLAComputeConfig,
    table: ComputeEnergyTablePJ,
) -> dict[str, float | str]:
    """Compare a conventional ANN with selectively replaced B-PLA operations."""

    bpla_mul = bpla_multiplier_energy_pj(config, table)["total_pj"]
    fp_gelu = fp32_gelu_energy_pj(table)["total_pj"]
    bpla_gelu = bpla_gelu_energy_pj(config, table)["total_pj"]
    exact_mul_sites = workload.multiply_sites - workload.bpla_multiply_sites
    exact_gelu_sites = workload.gelu_sites - workload.bpla_gelu_sites

    # One FP32 accumulation per scalar product is charged identically to both
    # variants. This is a MAC-equivalent convention and isolates product cost.
    ann_product = workload.multiply_sites * table.fp32_mul
    accumulation = workload.multiply_sites * table.fp32_add
    ann_gelu = workload.gelu_sites * fp_gelu
    bpla_product = (
        exact_mul_sites * table.fp32_mul
        + workload.bpla_multiply_sites * bpla_mul
    )
    variant_gelu = exact_gelu_sites * fp_gelu + workload.bpla_gelu_sites * bpla_gelu
    ann_softmax = fp32_softmax_energy_pj(workload.softmax_elements, workload.softmax_rows, table)["total_pj"]
    bpla_softmax = bpla_softmax_energy_pj(
        workload.bpla_softmax_elements,
        workload.bpla_softmax_rows,
        config,
        table,
    )["total_pj"]
    exact_softmax_elements = workload.softmax_elements - workload.bpla_softmax_elements
    exact_softmax_rows = workload.softmax_rows - workload.bpla_softmax_rows
    variant_softmax = (
        fp32_softmax_energy_pj(exact_softmax_elements, exact_softmax_rows, table)["total_pj"]
        + bpla_softmax
    )
    ann_layernorm = fp32_layernorm_energy_pj(
        workload.layernorm_elements,
        workload.layernorm_rows,
        table,
    )["total_pj"]
    bpla_layernorm = bpla_layernorm_energy_pj(
        workload.bpla_layernorm_elements,
        workload.bpla_layernorm_rows,
        config,
        table,
    )["total_pj"]
    exact_layernorm_elements = workload.layernorm_elements - workload.bpla_layernorm_elements
    exact_layernorm_rows = workload.layernorm_rows - workload.bpla_layernorm_rows
    variant_layernorm = (
        fp32_layernorm_energy_pj(exact_layernorm_elements, exact_layernorm_rows, table)["total_pj"]
        + bpla_layernorm
    )
    ann_total = ann_product + accumulation + ann_gelu + ann_softmax + ann_layernorm
    bpla_total = bpla_product + accumulation + variant_gelu + variant_softmax + variant_layernorm
    ratio = bpla_total / ann_total if ann_total else 1.0
    softmax_ratio = variant_softmax / ann_softmax if ann_softmax else 1.0
    layernorm_ratio = variant_layernorm / ann_layernorm if ann_layernorm else 1.0
    return {
        "label": workload.label,
        "multiply_sites": float(workload.multiply_sites),
        "bpla_multiply_sites": float(workload.bpla_multiply_sites),
        "gelu_sites": float(workload.gelu_sites),
        "bpla_gelu_sites": float(workload.bpla_gelu_sites),
        "softmax_rows": float(workload.softmax_rows),
        "softmax_elements": float(workload.softmax_elements),
        "bpla_softmax_rows": float(workload.bpla_softmax_rows),
        "layernorm_rows": float(workload.layernorm_rows),
        "layernorm_elements": float(workload.layernorm_elements),
        "bpla_layernorm_rows": float(workload.bpla_layernorm_rows),
        "fp32_mul_energy_pj": table.fp32_mul,
        "bpla_mul_energy_pj": bpla_mul,
        "fp32_gelu_energy_pj": fp_gelu,
        "bpla_gelu_energy_pj": bpla_gelu,
        "ann_product_pj": ann_product,
        "bpla_variant_product_pj": bpla_product,
        "common_accumulation_pj": accumulation,
        "ann_gelu_pj": ann_gelu,
        "bpla_variant_gelu_pj": variant_gelu,
        "ann_softmax_pj": ann_softmax,
        "bpla_variant_softmax_pj": variant_softmax,
        "softmax_ratio": softmax_ratio,
        "softmax_savings_pct": 100.0 * (1.0 - softmax_ratio),
        "ann_layernorm_pj": ann_layernorm,
        "bpla_variant_layernorm_pj": variant_layernorm,
        "layernorm_ratio": layernorm_ratio,
        "layernorm_savings_pct": 100.0 * (1.0 - layernorm_ratio),
        "ann_total_pj": ann_total,
        "bpla_total_pj": bpla_total,
        "bpla_over_ann": ratio,
        "savings_pct": 100.0 * (1.0 - ratio),
    }


def mlp_workload(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    linear_layers: int = 3,
    replace_linear: bool = True,
    replace_gelu: bool = True,
    max_linear_modules: int | None = None,
) -> ComputeWorkload:
    """Counts for the project's input->hidden->hidden->output toy MLP."""

    if linear_layers != 3:
        raise ValueError("the project toy MLP currently has exactly three linear layers.")
    layer_sites = [input_dim * hidden_dim, hidden_dim * hidden_dim, hidden_dim * output_dim]
    multiply_sites = sum(layer_sites)
    selected_layers = len(layer_sites) if max_linear_modules is None else max(0, min(max_linear_modules, len(layer_sites)))
    bpla_multiply_sites = sum(layer_sites[:selected_layers]) if replace_linear else 0
    gelu_sites = 2 * hidden_dim
    return ComputeWorkload(
        multiply_sites=multiply_sites,
        bpla_multiply_sites=bpla_multiply_sites,
        gelu_sites=gelu_sites,
        bpla_gelu_sites=gelu_sites if replace_gelu else 0,
        label="MNIST MLP / sample",
    )


def _is_bpla_linear(module: Any) -> bool:
    return module.__class__.__name__ in {"TorchBPLALinear", "TorchBPLAConv1D"}


def _is_linear(module: Any) -> bool:
    name = module.__class__.__name__
    return name in {"Linear", "Conv1D", "TorchBPLALinear", "TorchBPLAConv1D"}


def _linear_shape(module: Any) -> tuple[int, int]:
    weight = module.weight
    if module.__class__.__name__ in {"Conv1D", "TorchBPLAConv1D"}:
        return int(weight.shape[0]), int(weight.shape[1])
    return int(weight.shape[1]), int(weight.shape[0])


def _activation_module_counts(model: Any) -> tuple[int, int]:
    total = 0
    bpla = 0
    for module in model.modules():
        name = module.__class__.__name__.lower()
        if name == "torchbplaactivation" or "gelu" in name:
            total += 1
            if name == "torchbplaactivation":
                bpla += 1
    return total, bpla


def _layernorm_counts(model: Any, positions: int) -> tuple[int, int, int, int]:
    total_rows = total_elements = bpla_rows = bpla_elements = 0
    for module in model.modules():
        name = module.__class__.__name__
        if name not in {"LayerNorm", "TorchBPLALayerNorm"}:
            continue
        normalized_shape = getattr(module, "normalized_shape", ())
        width = 1
        for size in normalized_shape:
            width *= int(size)
        rows = positions
        elements = rows * width
        total_rows += rows
        total_elements += elements
        if name == "TorchBPLALayerNorm":
            bpla_rows += rows
            bpla_elements += elements
    return total_rows, total_elements, bpla_rows, bpla_elements


def vit_workload_from_model(
    model: Any,
    attention_mode: str = "bpla-full",
) -> ComputeWorkload:
    """Infer ViT multiplication/GELU counts from a converted model."""

    cfg = model.config
    image_size = cfg.image_size if isinstance(cfg.image_size, int) else cfg.image_size[0]
    patch_size = cfg.patch_size if isinstance(cfg.patch_size, int) else cfg.patch_size[0]
    patches = (int(image_size) // int(patch_size)) ** 2
    tokens = patches + 1
    total_mul = 0
    bpla_mul = 0
    for name, module in model.named_modules():
        if not _is_linear(module):
            continue
        in_features, out_features = _linear_shape(module)
        positions = 1 if name.endswith("classifier") else tokens
        sites = positions * in_features * out_features
        total_mul += sites
        if _is_bpla_linear(module):
            bpla_mul += sites

    # Patch embedding Conv2d is exact in the current conversion.
    total_mul += patches * int(cfg.num_channels) * patch_size * patch_size * int(cfg.hidden_size)

    layers = int(cfg.num_hidden_layers)
    qk_sites = layers * tokens * tokens * int(cfg.hidden_size)
    pv_sites = qk_sites
    total_mul += qk_sites + pv_sites
    if attention_mode in {"bpla-qk", "bpla-full"}:
        bpla_mul += qk_sites
    if attention_mode in {"bpla-pv", "bpla-full"}:
        bpla_mul += pv_sites

    activation_modules, bpla_activation_modules = _activation_module_counts(model)
    gelu_sites = activation_modules * tokens * int(cfg.intermediate_size)
    bpla_gelu_sites = bpla_activation_modules * tokens * int(cfg.intermediate_size)
    heads = int(cfg.num_attention_heads)
    softmax_rows = layers * heads * tokens
    softmax_elements = softmax_rows * tokens
    bpla_softmax = bool(getattr(model, "_bpla_softmax_enabled", False))
    ln_rows, ln_elements, bpla_ln_rows, bpla_ln_elements = _layernorm_counts(model, tokens)
    return ComputeWorkload(
        multiply_sites=total_mul,
        bpla_multiply_sites=bpla_mul,
        gelu_sites=gelu_sites,
        bpla_gelu_sites=bpla_gelu_sites,
        label=f"ViT / image (tokens={tokens})",
        softmax_rows=softmax_rows,
        softmax_elements=softmax_elements,
        bpla_softmax_rows=softmax_rows if bpla_softmax else 0,
        bpla_softmax_elements=softmax_elements if bpla_softmax else 0,
        layernorm_rows=ln_rows,
        layernorm_elements=ln_elements,
        bpla_layernorm_rows=bpla_ln_rows,
        bpla_layernorm_elements=bpla_ln_elements,
    )


def gpt2_workload_from_model(
    model: Any,
    sequence_length: int,
    attention_mode: str = "bpla-full",
) -> ComputeWorkload:
    """Infer GPT-2 prefill counts, including the exact LM head, from a model."""

    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive.")
    total_mul = 0
    bpla_mul = 0
    for _, module in model.named_modules():
        if not _is_linear(module):
            continue
        in_features, out_features = _linear_shape(module)
        sites = sequence_length * in_features * out_features
        total_mul += sites
        if _is_bpla_linear(module):
            bpla_mul += sites

    cfg = model.config
    layers = int(cfg.n_layer)
    hidden = int(cfg.n_embd)
    qk_sites = layers * sequence_length * sequence_length * hidden
    pv_sites = qk_sites
    total_mul += qk_sites + pv_sites
    if attention_mode in {"bpla-qk", "bpla-full"}:
        bpla_mul += qk_sites
    if attention_mode in {"bpla-pv", "bpla-full"}:
        bpla_mul += pv_sites

    activation_modules, bpla_activation_modules = _activation_module_counts(model)
    intermediate = int(getattr(cfg, "n_inner", None) or 4 * hidden)
    gelu_sites = activation_modules * sequence_length * intermediate
    bpla_gelu_sites = bpla_activation_modules * sequence_length * intermediate
    heads = int(cfg.n_head)
    softmax_rows = layers * heads * sequence_length
    softmax_elements = softmax_rows * sequence_length
    bpla_softmax = bool(getattr(model, "_bpla_softmax_enabled", False))
    ln_rows, ln_elements, bpla_ln_rows, bpla_ln_elements = _layernorm_counts(model, sequence_length)
    return ComputeWorkload(
        multiply_sites=total_mul,
        bpla_multiply_sites=bpla_mul,
        gelu_sites=gelu_sites,
        bpla_gelu_sites=bpla_gelu_sites,
        label=f"GPT-2 prefill / sequence (length={sequence_length})",
        softmax_rows=softmax_rows,
        softmax_elements=softmax_elements,
        bpla_softmax_rows=softmax_rows if bpla_softmax else 0,
        bpla_softmax_elements=softmax_elements if bpla_softmax else 0,
        layernorm_rows=ln_rows,
        layernorm_elements=ln_elements,
        bpla_layernorm_rows=bpla_ln_rows,
        bpla_layernorm_elements=bpla_ln_elements,
    )


def format_compute_energy_report(result: dict[str, float | str]) -> str:
    """Compact report shared by MLP, ViT, and GPT-2 probes."""

    return "\n".join(
        [
            "Compute-only theoretical energy (memory excluded)",
            f"scope                    : {result['label']}",
            "modeled operations       : products + accumulations + GELU + Softmax + LayerNorm",
            f"multiply sites           : {int(result['multiply_sites']):,}",
            f"B-PLA multiply sites     : {int(result['bpla_multiply_sites']):,}",
            f"GELU sites               : {int(result['gelu_sites']):,}",
            f"B-PLA GELU sites         : {int(result['bpla_gelu_sites']):,}",
            f"Softmax rows/elements    : {int(result['softmax_rows']):,} / {int(result['softmax_elements']):,}",
            f"B-PLA Softmax rows       : {int(result['bpla_softmax_rows']):,}",
            f"LayerNorm rows/elements  : {int(result['layernorm_rows']):,} / {int(result['layernorm_elements']):,}",
            f"B-PLA LayerNorm rows     : {int(result['bpla_layernorm_rows']):,}",
            f"FP32/B-PLA multiply      : {float(result['fp32_mul_energy_pj']):.6f} / {float(result['bpla_mul_energy_pj']):.6f} pJ",
            f"FP32/B-PLA GELU          : {float(result['fp32_gelu_energy_pj']):.6f} / {float(result['bpla_gelu_energy_pj']):.6f} pJ",
            f"ANN/variant Softmax      : {float(result['ann_softmax_pj']) / 1e6:.6f} / {float(result['bpla_variant_softmax_pj']) / 1e6:.6f} uJ",
            f"Softmax ratio/savings    : {float(result['softmax_ratio']):.6f}x / {float(result['softmax_savings_pct']):.2f}%",
            f"ANN/variant LayerNorm    : {float(result['ann_layernorm_pj']) / 1e6:.6f} / {float(result['bpla_variant_layernorm_pj']) / 1e6:.6f} uJ",
            f"LayerNorm ratio/savings  : {float(result['layernorm_ratio']):.6f}x / {float(result['layernorm_savings_pct']):.2f}%",
            f"ANN modeled energy       : {float(result['ann_total_pj']) / 1e6:.6f} uJ",
            f"B-PLA modeled energy     : {float(result['bpla_total_pj']) / 1e6:.6f} uJ",
            f"B-PLA / ANN              : {float(result['bpla_over_ann']):.6f}x",
            f"theoretical savings      : {float(result['savings_pct']):.2f}%",
            "exp/reciprocal/rsqrt baseline defaults to one FP32-multiply equivalent each.",
        ]
    )
