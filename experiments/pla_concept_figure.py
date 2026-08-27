"""
What piecewise linear approximation is, and what B-PLA does with it.

The introduction states the idea in one equation; this figure is what makes that
equation concrete before the reader reaches the method. Both panels are produced
by the shipped implementation rather than drawn by hand, so what the figure shows
is what the code computes.

Left -- the idea on a nonlinear operator
----------------------------------------
GELU, a deliberately coarse six-segment least-squares fit that shows what
Equation (1) does, and B-PLA's own approximation. The coarse fit is there
because B-PLA's is invisible: at k=4 it places 227 segments and lies on the
function, so a plot of it alone would say nothing about how it is built. The
contrast is the point -- the same construction, at a resolution the routing
makes free. B-PLA's segments are delimited by floating-point exponent and
mantissa-prefix boundaries rather than by stored breakpoints, so no comparator
chain selects them.

Right -- the same idea applied to multiplication
------------------------------------------------
The residual of the tile-wise planar fit to the mantissa interaction term
m1*m2. Fitting each tile at its centre makes the residual exactly
(m1-mu_i)(m2-nu_j), which vanishes along both centre lines and is bounded by
2^-(2k+2) -- the bright grid is that structure, not noise. Mitchell-family
multiplication is the special case that discards m1*m2 entirely; its error over
the same square would be the whole surface, up to 0.25, which the shared colour
scale would render uniformly saturated.

Usage
-----
    python experiments/pla_concept_figure.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from modules.torch_bpla import (
    SharedBPLATables,
    TorchBPLAConfig,
    activation_prefix_index_torch,
    bpla_activation_torch,
)

COLOR_EXACT = "#333333"
COLOR_BPLA = "#1f77b4"


def _segment_edges(x: torch.Tensor, index: torch.Tensor) -> list[float]:
    """Points where the router's segment index changes, read off the router."""

    changes = (index[1:] != index[:-1]).nonzero().flatten()
    return [float(x[i + 1]) for i in changes.tolist()]


def render(output: Path, prefix_bits: int, terms: int) -> Path:
    figure, (left, right) = plt.subplots(1, 2, figsize=(7.4, 3.1))

    # ------------------------------------------------------------------ left
    x = torch.linspace(-4.0, 4.0, 4001)
    exact = torch.nn.functional.gelu(x, approximate="tanh")

    # A genuine least-squares piecewise fit at a resolution the eye can resolve.
    # This is the idea of Equation (1), not B-PLA's router: B-PLA already places
    # 227 segments at k=4, and at that density -- or even at k=1, where the
    # exponent bins alone give 31 -- the approximation lies on the function and
    # a plot of it would show nothing about how it is built.
    breaks = torch.tensor([-4.0, -2.0, -1.0, -0.25, 0.5, 1.5, 4.0])
    coarse = torch.empty_like(exact)
    for lo, hi in zip(breaks[:-1], breaks[1:]):
        mask = (x >= lo) & (x <= hi)
        xs, ys = x[mask], exact[mask]
        design = torch.stack([xs, torch.ones_like(xs)], dim=1)
        slope, offset = torch.linalg.lstsq(design, ys.unsqueeze(1)).solution.flatten()
        coarse[mask] = slope * xs + offset

    config = TorchBPLAConfig(
        prefix_bits=prefix_bits, affine_path="dyadic", dyadic_terms=terms,
        nonlinear_dyadic_terms=4, activation_range=4.0,
    )
    table = SharedBPLATables(config).activation("gelu", x.device, x.dtype)
    approximate = bpla_activation_torch(x, table, config)
    index = activation_prefix_index_torch(
        x, config, int(table["min_e_routing"]), int(table["max_e_routing"])
    )
    edges = _segment_edges(x, index)

    for edge in breaks[1:-1]:
        left.axvline(float(edge), color="0.8", linewidth=0.8, linestyle=":", zorder=0)
    left.plot(x, exact, color=COLOR_EXACT, linewidth=2.4, label="GELU", zorder=2)
    left.plot(x, coarse, color="#d62728", linewidth=1.5, zorder=3,
              label="the idea: 6 affine segments")
    left.set_xlim(-3.0, 2.0)
    left.set_ylim(-0.55, 2.15)
    left.plot(x, approximate, color=COLOR_BPLA, linewidth=1.5, linestyle="--", zorder=4,
              label=f"B-PLA, $k$={prefix_bits} ({len(edges) + 1} segments)")
    left.scatter(breaks[1:-1], torch.nn.functional.gelu(breaks[1:-1], approximate="tanh"),
                 s=18, color="#d62728", zorder=5)
    left.set_xlabel("$x$")
    left.set_ylabel("$f(x)$")
    left.set_title("A nonlinear operator, segment by segment", fontsize=9)
    left.grid(alpha=0.2)
    left.legend(fontsize=6.8, loc="upper left", frameon=False)
    left.annotate(
        "B-PLA's curve is hidden under the function: max error "
        f"{float((approximate - exact).abs().max()):.0e}",
        xy=(0.5, 0.02), xycoords="axes fraction", ha="center", va="bottom",
        fontsize=6.6, color="0.35",
    )

    # ----------------------------------------------------------------- right
    # The residual of the tile-centre planar fit, computed the way the
    # multiplier computes it: (m1 - mu_i)(m2 - nu_j).
    segments = 1 << prefix_bits
    grid = np.linspace(0.0, 1.0, 512, endpoint=False)
    m1, m2 = np.meshgrid(grid, grid, indexing="ij")
    centres = (np.arange(segments) + 0.5) / segments
    mu = centres[np.clip((m1 * segments).astype(int), 0, segments - 1)]
    nu = centres[np.clip((m2 * segments).astype(int), 0, segments - 1)]
    residual = np.abs((m1 - mu) * (m2 - nu))

    image = right.imshow(
        residual.T, origin="lower", extent=(0, 1, 0, 1), cmap="magma",
        vmin=0.0, vmax=2.0 ** -(2 * prefix_bits + 2), aspect="equal",
    )
    for position in np.arange(1, segments) / segments:
        right.axvline(position, color="white", linewidth=0.25, alpha=0.4)
        right.axhline(position, color="white", linewidth=0.25, alpha=0.4)
    right.set_xlabel("$m_1$")
    right.set_ylabel("$m_2$")
    right.set_title("The term a multiplier has to approximate", fontsize=9)
    bar = figure.colorbar(image, ax=right, fraction=0.046, pad=0.03)
    bar.set_label(r"$|\,m_1m_2 - \widehat{m_1m_2}\,|$", fontsize=7.5)
    bar.ax.tick_params(labelsize=6.5)
    right.annotate(
        f"residual $=(m_1-\\mu_i)(m_2-\\nu_j)$,\nbounded by $2^{{-{2 * prefix_bits + 2}}}$",
        xy=(0.04, 0.93), xycoords="axes fraction", ha="left", va="top",
        fontsize=6.8, color="white",
    )

    figure.suptitle(
        "Piecewise linear approximation, routed by the floating-point fields",
        fontsize=9.5,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    figure.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(figure)

    peak = float(residual.max())
    bound = 2.0 ** -(2 * prefix_bits + 2)
    print(f"  coarse illustrative fit: 6 segments, max err "
          f"{float((coarse - exact).abs().max()):.3e}")
    print(f"  B-PLA k={prefix_bits}: {len(edges) + 1} segments")
    print(f"  peak tile residual {peak:.3e} against the bound {bound:.3e} "
          f"({'within' if peak <= bound * 1.001 else 'ABOVE'})")
    print(f"  GELU max |error|: {float((approximate - exact).abs().max()):.3e}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Render the PLA concept figure.")
    parser.add_argument("--prefix-bits", type=int, default=4)
    parser.add_argument("--dyadic-terms", type=int, default=2)
    parser.add_argument(
        "--output", type=Path,
        default=Path(__file__).resolve().parent / "fig_pla_concept.png",
    )
    args = parser.parse_args()
    print(f"wrote {render(args.output, args.prefix_bits, args.dyadic_terms)}")


if __name__ == "__main__":
    main()
