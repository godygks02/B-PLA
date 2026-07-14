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
    """Modeled multiplication, accumulation, and GELU sites."""

    multiply_sites: int
    bpla_multiply_sites: int
    gelu_sites: int
    bpla_gelu_sites: int
    label: str

    def __post_init__(self) -> None:
        values = (
            self.multiply_sites,
            self.bpla_multiply_sites,
            self.gelu_sites,
            self.bpla_gelu_sites,
        )
        if any(value < 0 for value in values):
            raise ValueError("workload counts must be non-negative.")
        if self.bpla_multiply_sites > self.multiply_sites:
            raise ValueError("B-PLA multiplication sites cannot exceed total sites.")
        if self.bpla_gelu_sites > self.gelu_sites:
            raise ValueError("B-PLA GELU sites cannot exceed total sites.")


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
    ann_total = ann_product + accumulation + ann_gelu
    bpla_total = bpla_product + accumulation + variant_gelu
    ratio = bpla_total / ann_total if ann_total else 1.0
    return {
        "label": workload.label,
        "multiply_sites": float(workload.multiply_sites),
        "bpla_multiply_sites": float(workload.bpla_multiply_sites),
        "gelu_sites": float(workload.gelu_sites),
        "bpla_gelu_sites": float(workload.bpla_gelu_sites),
        "fp32_mul_energy_pj": table.fp32_mul,
        "bpla_mul_energy_pj": bpla_mul,
        "fp32_gelu_energy_pj": fp_gelu,
        "bpla_gelu_energy_pj": bpla_gelu,
        "ann_product_pj": ann_product,
        "bpla_variant_product_pj": bpla_product,
        "common_accumulation_pj": accumulation,
        "ann_gelu_pj": ann_gelu,
        "bpla_variant_gelu_pj": variant_gelu,
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
    return ComputeWorkload(
        multiply_sites=total_mul,
        bpla_multiply_sites=bpla_mul,
        gelu_sites=gelu_sites,
        bpla_gelu_sites=bpla_gelu_sites,
        label=f"ViT / image (tokens={tokens})",
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
    return ComputeWorkload(
        multiply_sites=total_mul,
        bpla_multiply_sites=bpla_mul,
        gelu_sites=gelu_sites,
        bpla_gelu_sites=bpla_gelu_sites,
        label=f"GPT-2 prefill / sequence (length={sequence_length})",
    )


def format_compute_energy_report(result: dict[str, float | str]) -> str:
    """Compact report shared by MLP, ViT, and GPT-2 probes."""

    return "\n".join(
        [
            "Compute-only theoretical energy (memory excluded)",
            f"scope                    : {result['label']}",
            "modeled operations       : products + common accumulations + GELU",
            f"multiply sites           : {int(result['multiply_sites']):,}",
            f"B-PLA multiply sites     : {int(result['bpla_multiply_sites']):,}",
            f"GELU sites               : {int(result['gelu_sites']):,}",
            f"B-PLA GELU sites         : {int(result['bpla_gelu_sites']):,}",
            f"FP32/B-PLA multiply      : {float(result['fp32_mul_energy_pj']):.6f} / {float(result['bpla_mul_energy_pj']):.6f} pJ",
            f"FP32/B-PLA GELU          : {float(result['fp32_gelu_energy_pj']):.6f} / {float(result['bpla_gelu_energy_pj']):.6f} pJ",
            f"ANN modeled energy       : {float(result['ann_total_pj']) / 1e6:.6f} uJ",
            f"B-PLA modeled energy     : {float(result['bpla_total_pj']) / 1e6:.6f} uJ",
            f"B-PLA / ANN              : {float(result['bpla_over_ann']):.6f}x",
            f"theoretical savings      : {float(result['savings_pct']):.2f}%",
        ]
    )
