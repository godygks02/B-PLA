"""
Primitive-level comparison of PAO/PAM against B-PLA multiplication.

Answers the first question of the submission plan: at the level of a single
scalar product, how much accuracy does the fitted mantissa-interaction plane
buy over the plain piecewise affine multiplication of Kosson and Jaggi, and
what does it cost in coefficient storage and shift-add terms?

Nothing here is an energy or area claim. The cost columns are arithmetic and
storage proxies only; a physical comparison needs synthesis.

Outputs (written next to this file unless --output-dir is given):
    pao_vs_bpla_primitive.json   full record incl. configuration and metrics
    pao_vs_bpla_primitive.csv    one row per method/configuration
    fig_multiplier_error_map.png mantissa-plane relative error maps
    fig_multiplier_pareto.png    accuracy vs. shift-add cost proxy
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from modules.torch_bpla import TorchBPLAConfig, bpla_multiply_torch
from modules.torch_pao import TorchPAOConfig, pao_multiply_torch


# Bits per dyadic term: one sign bit plus the shift index. ``max_shift`` of 16
# needs 5 bits to index shifts 0..16, so a term costs 6 bits.
_SIGN_BITS = 1


@dataclass
class MethodResult:
    method: str
    prefix_bits: int | None
    dyadic_terms: int | None
    distribution: str
    mae: float
    rmse: float
    mean_relative_error: float
    p99_abs_relative_error: float
    max_abs_relative_error: float
    coefficient_entries: int
    coefficient_bits: int
    shift_add_terms_per_product: int
    int_adds_per_product: int
    fp_mults_per_product: int


def _relative_error(approx: torch.Tensor, exact: torch.Tensor) -> torch.Tensor:
    """Signed relative error. The sign matters: PAM only ever underestimates
    the magnitude, so its error is a bias that accumulates along a dot product,
    whereas the B-PLA tile residual is close to zero-mean and partly cancels."""

    tiny = torch.finfo(exact.dtype).tiny
    safe = torch.where(exact.abs() < tiny, torch.full_like(exact, tiny), exact)
    return (approx - exact) / safe


def _cost_proxy(
    method: str,
    prefix_bits: int | None,
    dyadic_terms: int | None,
    max_shift: int,
    separable: bool = True,
) -> dict[str, int]:
    """Per-product arithmetic and per-table storage proxies.

    PAM: two int32 additions (operand add plus exponent-bias correction).
    B-PLA float: three float coefficients read per product, two float
    multiplies and three adds to evaluate ``a*m1 + b*m2 + c`` and fold it in.
    B-PLA dyadic: the two slope products become ``T`` shift-add terms each and
    the offset is a stored constant, so no float multiplier remains on the
    mantissa path.
    """

    if method in {"pao", "pao-alpha"}:
        # The alpha correction is itself one more PAM, so two more int adds.
        return {
            "coefficient_entries": 0,
            "coefficient_bits": 0,
            "shift_add_terms_per_product": 0,
            "int_adds_per_product": 2 if method == "pao" else 4,
            "fp_mults_per_product": 0,
        }

    assert prefix_bits is not None
    # The separable form stores only the 2^k tile centres; the legacy plane form
    # stored three 2^k x 2^k coefficient planes generated from exactly that
    # array, so its table was redundant by a factor of 3 * 2^k.
    entries = (1 << prefix_bits) if separable else 3 * (1 << (2 * prefix_bits))
    if method == "bpla-float":
        return {
            "coefficient_entries": entries,
            "coefficient_bits": entries * 32,
            "shift_add_terms_per_product": 0,
            "int_adds_per_product": 1,  # exponent sum
            "fp_mults_per_product": 2,
        }

    assert dyadic_terms is not None
    shift_index_bits = max(1, (max_shift).bit_length())
    term_bits = _SIGN_BITS + shift_index_bits
    return {
        "coefficient_entries": entries,
        "coefficient_bits": entries * dyadic_terms * term_bits,
        # Two coefficient-operand products, each expanded into T signed
        # power-of-two terms. The separable form needs no offset coefficient,
        # so it also saves one addition per product.
        "shift_add_terms_per_product": 2 * dyadic_terms,
        "int_adds_per_product": 1,
        "fp_mults_per_product": 0,
    }


def _sample_operands(distribution: str, count: int, generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """Operand pairs for the three regimes the paper needs to distinguish."""

    if distribution == "uniform":
        a = torch.empty(count).uniform_(-6.0, 6.0, generator=generator)
        b = torch.empty(count).uniform_(-6.0, 6.0, generator=generator)
    elif distribution == "log-uniform":
        # Spread across octaves so no single exponent range dominates.
        exponent_a = torch.empty(count).uniform_(-8.0, 8.0, generator=generator)
        exponent_b = torch.empty(count).uniform_(-8.0, 8.0, generator=generator)
        sign_a = torch.randint(0, 2, (count,), generator=generator) * 2 - 1
        sign_b = torch.randint(0, 2, (count,), generator=generator) * 2 - 1
        a = sign_a * torch.exp2(exponent_a)
        b = sign_b * torch.exp2(exponent_b)
    elif distribution == "activation-weight":
        # Coarse stand-in for a Transformer product: unit-scale activation
        # against a small-scale weight.
        a = torch.randn(count, generator=generator)
        b = torch.randn(count, generator=generator) * 0.02
    else:
        raise ValueError(f"Unknown distribution {distribution!r}.")
    nonzero = (a != 0) & (b != 0)
    return a[nonzero].to(torch.float32), b[nonzero].to(torch.float32)


def _fit_alpha(
    a: torch.Tensor, b: torch.Tensor, exact: torch.Tensor, args: argparse.Namespace
) -> float:
    """Pick the alpha that makes the PAM gain unbiased on this distribution.

    This is the most favourable setting for the baseline: alpha is fitted on the
    same data it is evaluated on, using the exact products the method would not
    normally have. Anything B-PLA still wins under this handicap is not an
    artifact of leaving the baseline uncorrected.
    """

    best_alpha, best_error = 1.0, float("inf")
    for candidate in torch.linspace(1.0, 1.12, 61).tolist():
        approx = pao_multiply_torch(a, b, TorchPAOConfig(alpha=candidate)).to(torch.float64)
        reference = exact.to(torch.float64)
        gain = float((approx * reference).sum() / reference.pow(2).sum())
        if abs(gain - 1.0) < best_error:
            best_alpha, best_error = candidate, abs(gain - 1.0)
    return best_alpha


def evaluate_methods(args: argparse.Namespace) -> list[MethodResult]:
    generator = torch.Generator().manual_seed(args.seed)
    results: list[MethodResult] = []

    for distribution in args.distributions:
        a, b = _sample_operands(distribution, args.num_samples, generator)
        exact = (a.to(torch.float64) * b.to(torch.float64)).to(torch.float32)

        configurations: list[tuple[str, int | None, int | None, torch.Tensor]] = [
            ("pao", None, None, pao_multiply_torch(a, b, TorchPAOConfig())),
            # Section 2.7 of Kosson and Jaggi sketches a single-constant
            # correction x1*x2*alpha but reports no results for it. It removes
            # the systematic contraction, so the comparison is not honest
            # without it: we fit alpha to make the gain unbiased and report the
            # corrected operation as its own condition.
            (
                "pao-alpha",
                None,
                None,
                pao_multiply_torch(a, b, TorchPAOConfig(alpha=_fit_alpha(a, b, exact, args))),
            ),
        ]
        for prefix_bits in args.prefix_bits:
            float_config = TorchBPLAConfig(
                prefix_bits=prefix_bits,
                affine_path="float",
                multiplier_form=args.multiplier_form,
            )
            configurations.append(
                ("bpla-float", prefix_bits, None, bpla_multiply_torch(a, b, float_config))
            )
            for terms in args.dyadic_terms:
                dyadic_config = TorchBPLAConfig(
                    prefix_bits=prefix_bits,
                    affine_path="dyadic",
                    dyadic_terms=terms,
                    max_shift=args.max_shift,
                    multiplier_form=args.multiplier_form,
                )
                configurations.append(
                    ("bpla-dyadic", prefix_bits, terms, bpla_multiply_torch(a, b, dyadic_config))
                )

        for method, prefix_bits, terms, approx in configurations:
            error = approx - exact
            relative = _relative_error(approx, exact)
            absolute_relative = relative.abs()
            cost = _cost_proxy(
                method, prefix_bits, terms, args.max_shift, args.multiplier_form == "separable"
            )
            results.append(
                MethodResult(
                    method=method,
                    prefix_bits=prefix_bits,
                    dyadic_terms=terms,
                    distribution=distribution,
                    mae=float(error.abs().mean()),
                    rmse=float(error.pow(2).mean().sqrt()),
                    mean_relative_error=float(relative.mean()),
                    p99_abs_relative_error=float(torch.quantile(absolute_relative, 0.99)),
                    max_abs_relative_error=float(absolute_relative.max()),
                    **cost,
                )
            )
    return results


def mantissa_error_map(method: str, prefix_bits: int, terms: int, resolution: int, max_shift: int) -> torch.Tensor:
    """Relative error over one mantissa octave, the domain Figure 2 of PAO uses."""

    grid = (torch.arange(resolution, dtype=torch.float32) + 0.5) / resolution
    m1 = grid[:, None].expand(resolution, resolution).contiguous()
    m2 = grid[None, :].expand(resolution, resolution).contiguous()
    a = 1.0 + m1
    b = 1.0 + m2
    exact = a.to(torch.float64) * b.to(torch.float64)

    if method == "pao":
        approx = pao_multiply_torch(a, b, TorchPAOConfig())
    elif method == "bpla-float":
        approx = bpla_multiply_torch(a, b, TorchBPLAConfig(prefix_bits=prefix_bits, affine_path="float"))
    else:
        approx = bpla_multiply_torch(
            a,
            b,
            TorchBPLAConfig(
                prefix_bits=prefix_bits,
                affine_path="dyadic",
                dyadic_terms=terms,
                max_shift=max_shift,
            ),
        )
    return ((approx.to(torch.float64) - exact) / exact * 100.0).to(torch.float32)


def accumulation_sweep(args: argparse.Namespace) -> list[dict[str, float | int | str]]:
    """Decompose the error of a length-K dot product into gain and residual.

    The per-product error statistics do not by themselves predict what a
    Transformer sees, because a dot product accumulates K of them and the sum
    is itself near zero-mean, so a plain relative error is not well defined.

    We instead fit the accumulated output against the exact one over many
    trials. ``gain`` is the least-squares slope: PAM can only shrink a
    magnitude, so every product is scaled by roughly ``1 - 0.038`` and the whole
    dot product inherits that contraction no matter how long it is. ``residual``
    is the RMS of what the gain does not explain, normalised by the RMS of the
    exact sum. A systematic gain error compounds across layers; a zero-mean
    residual averages down.
    """

    generator = torch.Generator().manual_seed(args.seed + 1)
    rows: list[dict[str, float | int | str]] = []
    for length in args.accumulation_lengths:
        a = torch.randn(args.accumulation_trials, length, generator=generator)
        b = torch.randn(args.accumulation_trials, length, generator=generator) * 0.02
        exact = (a.to(torch.float64) * b.to(torch.float64)).sum(dim=-1)

        variants: list[tuple[str, int | None, int | None, torch.Tensor]] = [
            ("pao", None, None, pao_multiply_torch(a, b, TorchPAOConfig()).sum(dim=-1)),
            (
                "pao-alpha",
                None,
                None,
                pao_multiply_torch(
                    a, b, TorchPAOConfig(alpha=args.accumulation_alpha)
                ).sum(dim=-1),
            ),
        ]
        for prefix_bits in args.figure_prefix_bits:
            variants.append(
                (
                    "bpla-float",
                    prefix_bits,
                    None,
                    bpla_multiply_torch(
                        a, b, TorchBPLAConfig(prefix_bits=prefix_bits, affine_path="float")
                    ).sum(dim=-1),
                )
            )
            variants.append(
                (
                    "bpla-dyadic",
                    prefix_bits,
                    args.figure_dyadic_terms,
                    bpla_multiply_torch(
                        a,
                        b,
                        TorchBPLAConfig(
                            prefix_bits=prefix_bits,
                            affine_path="dyadic",
                            dyadic_terms=args.figure_dyadic_terms,
                            max_shift=args.max_shift,
                        ),
                    ).sum(dim=-1),
                )
            )

        exact_energy = exact.pow(2).sum()
        exact_rms = exact.pow(2).mean().sqrt()
        for method, prefix_bits, terms, approx in variants:
            approx = approx.to(torch.float64)
            gain = float((approx * exact).sum() / exact_energy)
            residual = float((approx - gain * exact).pow(2).mean().sqrt() / exact_rms)
            rows.append(
                {
                    "method": method,
                    "prefix_bits": prefix_bits if prefix_bits is not None else "",
                    "dyadic_terms": terms if terms is not None else "",
                    "length": length,
                    "gain": gain,
                    "gain_error": gain - 1.0,
                    "normalized_residual_rms": residual,
                }
            )
    return rows


def write_accumulation_figure(
    args: argparse.Namespace, rows: list[dict[str, float | int | str]], output_dir: Path
) -> Path | None:
    """Both metrics turn out to be independent of K, so plotting against K
    would waste a panel. What matters instead is that a length-independent
    gain error compounds once matmuls are chained through a network."""

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    reference_length = max(args.accumulation_lengths)
    at_length = [r for r in rows if r["length"] == reference_length]

    def _label(row: dict[str, float | int | str]) -> str:
        if row["method"] == "pao":
            return "PAM"
        if row["method"] == "pao-alpha":
            return "PAM\n+$\\alpha$"
        suffix = "float" if row["method"] == "bpla-float" else f"dyadic $T$={row['dyadic_terms']}"
        return f"B-PLA {suffix}\n$k$={row['prefix_bits']}"

    labels = [_label(r) for r in at_length]
    gain_errors = [abs(float(r["gain_error"])) * 100.0 for r in at_length]
    palette = {"pao": "#d62728", "pao-alpha": "#ff7f0e", "bpla-float": "#1f77b4"}
    colors = [palette.get(str(r["method"]), "#2ca02c") for r in at_length]

    figure, (left, right) = plt.subplots(1, 2, figsize=(9.6, 3.9), constrained_layout=True)

    bars = left.bar(range(len(labels)), gain_errors, color=colors)
    for bar, value in zip(bars, gain_errors):
        left.annotate(
            f"{value:.3g}",
            (bar.get_x() + bar.get_width() / 2, value),
            ha="center",
            va="bottom",
            fontsize=7,
        )
    left.set_yscale("log")
    left.set_xticks(range(len(labels)))
    left.set_xticklabels(labels, fontsize=6.5, rotation=30, ha="right")
    left.set_ylabel("|gain $-$ 1| (%)")
    left.set_title(
        f"Gain error on a length-{reference_length} dot product", fontsize=10
    )
    left.grid(alpha=0.3, axis="y")

    depths = torch.arange(0, 49)
    for row, color in zip(at_length, colors):
        gain = float(row["gain"])
        style = "-" if str(row["method"]).startswith("pao") else "--"
        width = 2.0 if row["method"] == "pao" else 1.1
        right.plot(
            depths.numpy(),
            (gain ** depths.to(torch.float64)).numpy(),
            style,
            color=color,
            linewidth=width,
            label=_label(row).replace("\n", " "),
        )
    right.axhline(1.0, color="black", linewidth=0.6, alpha=0.5)
    right.set_xlabel("chained approximate matmuls")
    right.set_ylabel("cumulative activation scale")
    # A real Transformer does not compound freely: LayerNorm resets the scale
    # every block and the residual stream carries an uncontracted identity
    # path, so this curve is an upper bound on the effect, not a prediction.
    right.annotate(
        "upper bound: LayerNorm and the residual\nstream absorb most of this in practice",
        xy=(0.5, 0.30),
        xycoords="axes fraction",
        fontsize=6.5,
        color="0.35",
        ha="center",
    )
    right.set_title(
        "Unmitigated compounding (no renormalisation)", fontsize=10
    )
    right.grid(alpha=0.3)
    right.legend(fontsize=6.5, loc="lower left")

    figure.suptitle(
        "PAM contracts every dot product by a fixed factor; a fitted $\\alpha$ removes it",
        fontsize=11,
    )
    path = output_dir / "fig_accumulation_bias.png"
    figure.savefig(path, dpi=200)
    plt.close(figure)
    return path


def write_figures(args: argparse.Namespace, results: list[MethodResult], output_dir: Path) -> list[Path]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib unavailable; skipping figures.")
        return []

    written: list[Path] = []

    panels = [("pao", None, None, "PAM (Kosson & Jaggi)")]
    for prefix_bits in args.figure_prefix_bits:
        panels.append(("bpla-float", prefix_bits, None, f"B-PLA float, $k$={prefix_bits}"))
    panels.append(
        (
            "bpla-dyadic",
            args.figure_prefix_bits[-1],
            args.figure_dyadic_terms,
            f"B-PLA dyadic, $k$={args.figure_prefix_bits[-1]}, $T$={args.figure_dyadic_terms}",
        )
    )

    maps = [
        (title, mantissa_error_map(method, prefix_bits or 1, terms or 1, args.map_resolution, args.max_shift))
        for method, prefix_bits, terms, title in panels
    ]
    limit = max(float(m.abs().max()) for _, m in maps)
    # A shared linear scale would render every B-PLA panel blank next to PAM's
    # 11% band, so the panels share one symmetric-log scale instead.
    norm = matplotlib.colors.SymLogNorm(linthresh=0.01, vmin=-limit, vmax=limit, base=10)

    figure, axes = plt.subplots(1, len(maps), figsize=(3.1 * len(maps), 3.3), constrained_layout=True)
    axes = axes if hasattr(axes, "__len__") else [axes]
    for axis, (title, error_map) in zip(axes, maps):
        image = axis.imshow(
            error_map.numpy(),
            origin="lower",
            extent=(1.0, 2.0, 1.0, 2.0),
            cmap="RdBu_r",
            norm=norm,
        )
        peak = float(error_map.abs().max())
        axis.set_title(f"{title}\nmax |rel. err| = {peak:.3g}%", fontsize=9)
        axis.set_xlabel("$x_2$")
    axes[0].set_ylabel("$x_1$")
    figure.colorbar(image, ax=axes, shrink=0.85, label="relative error (%)")
    path = output_dir / "fig_multiplier_error_map.png"
    figure.savefig(path, dpi=200)
    plt.close(figure)
    written.append(path)

    distribution = args.distributions[0]
    subset = [r for r in results if r.distribution == distribution]
    pam = next(r for r in subset if r.method == "pao")
    pam_alpha = next((r for r in subset if r.method == "pao-alpha"), None)
    colors = plt.get_cmap("viridis")

    figure, (left, right) = plt.subplots(1, 2, figsize=(9.4, 3.9), constrained_layout=True)

    # Left: arithmetic cost. The float path is deliberately absent because it
    # still needs two float multiplies; it is drawn as the k-wise accuracy
    # floor that the dyadic path converges to.
    for index, prefix_bits in enumerate(args.prefix_bits):
        points = sorted(
            (r for r in subset if r.method == "bpla-dyadic" and r.prefix_bits == prefix_bits),
            key=lambda r: r.dyadic_terms or 0,
        )
        color = colors(index / max(1, len(args.prefix_bits) - 1))
        left.plot(
            [r.shift_add_terms_per_product for r in points],
            [r.p99_abs_relative_error * 100.0 for r in points],
            marker="o",
            markersize=4,
            color=color,
            label=f"$k$={prefix_bits}",
        )
        floor = next(
            (r for r in subset if r.method == "bpla-float" and r.prefix_bits == prefix_bits), None
        )
        if floor is not None:
            left.axhline(
                floor.p99_abs_relative_error * 100.0,
                color=color,
                linestyle=":",
                linewidth=0.8,
                alpha=0.7,
            )
    # Both baselines sit on the same axis: alpha compensation is one more PAM,
    # so it costs two more integer additions.
    for row, color, label in (
        (pam, "#d62728", "PAM"),
        (pam_alpha, "#ff7f0e", "PAM $+\\alpha$"),
    ):
        if row is None:
            continue
        left.scatter(
            [row.int_adds_per_product],
            [row.p99_abs_relative_error * 100.0],
            marker="*",
            s=190,
            color=color,
            zorder=5,
            label=f"{label} ({row.int_adds_per_product} int adds)",
        )
    left.set_yscale("log")
    left.set_xlabel("shift-add terms per scalar product ($2T$)")
    left.set_ylabel("p99 |relative error| (%)")
    left.set_title("Accuracy vs. arithmetic cost proxy", fontsize=10)
    left.grid(alpha=0.3)
    left.legend(fontsize=7, ncol=2)

    # Right: storage cost. Shows that widening the prefix stops paying once the
    # dyadic term budget, not the tile width, is the binding constraint.
    for index, terms in enumerate(args.dyadic_terms):
        points = sorted(
            (r for r in subset if r.method == "bpla-dyadic" and r.dyadic_terms == terms),
            key=lambda r: r.prefix_bits or 0,
        )
        right.plot(
            [r.coefficient_bits / 8192.0 for r in points],
            [r.p99_abs_relative_error * 100.0 for r in points],
            marker="s",
            markersize=4,
            color=colors(index / max(1, len(args.dyadic_terms) - 1)),
            label=f"$T$={terms}",
        )
    for row, color, label in (
        (pam, "#d62728", "PAM"),
        (pam_alpha, "#ff7f0e", "PAM $+\\alpha$"),
    ):
        if row is None:
            continue
        right.axhline(
            row.p99_abs_relative_error * 100.0,
            color=color,
            linestyle="--",
            linewidth=1.2,
            label=f"{label} (no table)",
        )
    right.set_xscale("log")
    right.set_yscale("log")
    right.set_xlabel("coefficient table size (KiB)")
    right.set_ylabel("p99 |relative error| (%)")
    right.set_title("Accuracy vs. coefficient storage", fontsize=10)
    right.grid(alpha=0.3)
    right.legend(fontsize=7, ncol=2)

    figure.suptitle(f"B-PLA multiplier configuration sweep ({distribution} operands)", fontsize=11)
    path = output_dir / "fig_multiplier_pareto.png"
    figure.savefig(path, dpi=200)
    plt.close(figure)
    written.append(path)
    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PAO/PAM vs. B-PLA primitive comparison.")
    parser.add_argument("--num-samples", type=int, default=400000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prefix-bits", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6])
    parser.add_argument("--dyadic-terms", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--max-shift", type=int, default=16)
    parser.add_argument("--multiplier-form", choices=["separable", "plane"], default="separable")
    parser.add_argument(
        "--distributions",
        nargs="+",
        default=["uniform", "log-uniform", "activation-weight"],
    )
    parser.add_argument("--figure-prefix-bits", type=int, nargs="+", default=[2, 4])
    parser.add_argument("--figure-dyadic-terms", type=int, default=2)
    parser.add_argument("--map-resolution", type=int, default=256)
    parser.add_argument("--accumulation-lengths", type=int, nargs="+", default=[1, 4, 16, 64, 256, 768, 3072])
    parser.add_argument("--accumulation-trials", type=int, default=4096)
    parser.add_argument(
        "--accumulation-alpha",
        type=float,
        default=1.056,
        help="PAM error-compensation constant fitted for an unbiased gain (Sec. 2.7).",
    )
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--no-figures", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    results = evaluate_methods(args)

    rows = [asdict(r) for r in results]
    csv_path = output_dir / "pao_vs_bpla_primitive.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    accumulation = accumulation_sweep(args)
    accumulation_path = output_dir / "pao_vs_bpla_accumulation.csv"
    with accumulation_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(accumulation[0].keys()))
        writer.writeheader()
        writer.writerows(accumulation)

    figures = [] if args.no_figures else write_figures(args, results, output_dir)
    if not args.no_figures:
        accumulation_figure = write_accumulation_figure(args, accumulation, output_dir)
        if accumulation_figure is not None:
            figures.append(accumulation_figure)

    record = {
        "configuration": {
            "num_samples": args.num_samples,
            "seed": args.seed,
            "prefix_bits": args.prefix_bits,
            "dyadic_terms": args.dyadic_terms,
            "max_shift": args.max_shift,
            "distributions": args.distributions,
        },
        "notes": [
            "PAM is the forward multiplication of Kosson and Jaggi (NeurIPS 2023), "
            "verified in tests/test_pao.py against the Mogami int-addition trick.",
            "Cost columns are arithmetic and storage proxies, not energy or area.",
            "No weight updates or approximation-aware training are involved.",
        ],
        "results": rows,
        "accumulation": accumulation,
        "figures": [str(p.name) for p in figures],
    }
    json_path = output_dir / "pao_vs_bpla_primitive.json"
    json_path.write_text(json.dumps(record, indent=2), encoding="utf-8")

    print(f"wrote {csv_path}")
    print(f"wrote {accumulation_path}")
    print(f"wrote {json_path}")
    for path in figures:
        print(f"wrote {path}")

    print()
    header = f"{'method':<14}{'k':>3}{'T':>3}  {'distribution':<19}{'MAE':>12}{'p99 |rel|':>12}{'max |rel|':>12}{'coef bits':>11}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r.method:<14}{r.prefix_bits if r.prefix_bits is not None else '-':>3}"
            f"{r.dyadic_terms if r.dyadic_terms is not None else '-':>3}  "
            f"{r.distribution:<19}{r.mae:>12.4e}{r.p99_abs_relative_error:>12.4e}"
            f"{r.max_abs_relative_error:>12.4e}{r.coefficient_bits:>11,}"
        )


if __name__ == "__main__":
    main()
