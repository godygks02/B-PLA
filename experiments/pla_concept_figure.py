"""Render the two-panel PLA concept figure used in the Introduction.

Panel (a) compares exact GELU with a deliberately coarse piecewise-affine fit,
plus an inset of the residual so the per-segment structure is unambiguous. Four
segments are used rather than a realistic count because the figure has to show
*how* the approximation is built: B-PLA's own GELU table places 227 segments and
its curve lies on the function, so plotting it would show nothing.

Panel (b) shows the exact mantissa-interaction surface ``m1*m2`` as a smooth
surface coloured by height, with the B-PLA tile-centre affine approximation
overlaid as 4x4 prefix-routed planes. The planes meet the surface along each
tile's centre lines, which is why the residual is exactly
``(m1 - mu_i)(m2 - nu_j)`` and is bounded by ``2^-(2k+2)``.

The default 7.2-inch width is suitable for a two-column paper figure. The
script writes a vector PDF, an editable SVG, and a 600-dpi PNG preview.

Usage
-----
    python experiments/pla_concept_figure.py
    python experiments/pla_concept_figure.py --prefix-bits 3
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np


# Okabe--Ito-compatible colours, with charcoal reserved for exact quantities.
COLOR_EXACT = "#262626"
COLOR_APPROX = "#0072B2"
COLOR_TILE = "#D55E00"
COLOR_BOUNDARY = "#B8B8B8"
HEIGHT_CMAP = "viridis"


def gelu(x: np.ndarray) -> np.ndarray:
    """Exact GELU, x * Phi(x), without requiring SciPy."""

    erf = np.fromiter(
        (math.erf(float(value) / math.sqrt(2.0)) for value in x),
        dtype=float,
        count=x.size,
    )
    return 0.5 * x * (1.0 + erf)


def fit_piecewise_affine(
    x: np.ndarray, y: np.ndarray, breakpoints: np.ndarray
) -> tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray]]]:
    """Fit one least-squares affine function independently on every segment."""

    approximate = np.empty_like(y)
    pieces: list[tuple[np.ndarray, np.ndarray]] = []
    for segment, (lo, hi) in enumerate(zip(breakpoints[:-1], breakpoints[1:])):
        # Make the half-open assignment explicit so every sample has one route.
        mask = (x >= lo) & (x < hi)
        if segment == len(breakpoints) - 2:
            mask = (x >= lo) & (x <= hi)
        xs, ys = x[mask], y[mask]
        slope, offset = np.polyfit(xs, ys, deg=1)
        fitted = slope * xs + offset
        approximate[mask] = fitted
        pieces.append((xs, fitted))
    return approximate, pieces


def configure_matplotlib() -> None:
    """Use compact typography that remains legible after paper scaling."""

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif"],
            "mathtext.fontset": "dejavuserif",
            "font.size": 8.5,
            "axes.titlesize": 9.2,
            "axes.labelsize": 9.0,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.4,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.major.size": 3.2,
            "ytick.major.size": 3.2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def draw_gelu_panel(axis: plt.Axes) -> float:
    """Draw exact GELU against a coarse PLA whose segments are visible."""

    x = np.linspace(-3.0, 3.0, 2401)
    exact = gelu(x)
    # Four segments, not a realistic count. At any density B-PLA actually uses,
    # the approximation lies on the function and the construction disappears.
    breakpoints = np.array([-3.0, -1.2, 0.0, 1.2, 3.0])
    approximate, pieces = fit_piecewise_affine(x, exact, breakpoints)

    for boundary in breakpoints[1:-1]:
        axis.axvline(
            boundary,
            color=COLOR_BOUNDARY,
            linewidth=0.7,
            linestyle=(0, (1.5, 2.2)),
            zorder=0,
        )

    axis.plot(x, exact, color=COLOR_EXACT, linewidth=2.0, label="Exact GELU", zorder=2)
    # Each piece is drawn on its own: two independently fitted segments are not
    # continuous at the boundary, and pretending otherwise would misdraw the
    # method.
    for index, (xs, fitted) in enumerate(pieces):
        axis.plot(
            xs,
            fitted,
            color=COLOR_APPROX,
            linewidth=1.8,
            label="Piecewise-affine fit" if index == 0 else None,
            zorder=3,
        )
        axis.plot(
            [xs[0], xs[-1]],
            [fitted[0], fitted[-1]],
            linestyle="none",
            marker="o",
            markersize=3.2,
            markerfacecolor="white",
            markeredgecolor=COLOR_APPROX,
            markeredgewidth=1.0,
            zorder=4,
        )

    axis.axhline(0.0, color="#777777", linewidth=0.55, zorder=0)
    axis.set(xlim=(-3.0, 3.0), ylim=(-0.62, 3.15), xlabel=r"Input $x$", ylabel="Output")
    axis.set_xticks(np.arange(-3, 4, 1))
    axis.set_yticks(np.arange(-0.5, 3.1, 0.5))
    axis.grid(True, color="#E5E5E5", linewidth=0.55)
    axis.set_axisbelow(True)
    axis.legend(loc="lower right", frameon=False, handlelength=2.2)
    axis.set_title("(a) A nonlinear operator", pad=7, fontweight="bold")

    # The residual makes the segmentation unambiguous: it returns toward zero
    # inside every segment and jumps at the boundaries.
    # Upper-left is the one region the curve never enters.
    inset = axis.inset_axes((0.075, 0.60, 0.40, 0.30))
    inset.plot(x, approximate - exact, color=COLOR_APPROX, linewidth=0.9)
    inset.axhline(0.0, color="#777777", linewidth=0.5)
    for boundary in breakpoints[1:-1]:
        inset.axvline(boundary, color=COLOR_BOUNDARY, linewidth=0.6,
                      linestyle=(0, (1.5, 2.2)))
    inset.set_xlim(-3.0, 3.0)
    inset.set_xticks([])
    inset.tick_params(axis="y", labelsize=5.6, pad=1.2, length=2.0)
    inset.set_title("residual", fontsize=6.0, pad=2.0)
    for spine in inset.spines.values():
        spine.set_linewidth(0.6)

    return float(np.max(np.abs(approximate - exact)))


def draw_mantissa_panel(axis: plt.Axes, prefix_bits: int):
    """Draw m1*m2 as a height-coloured surface under its prefix-tiled planes."""

    segments = 1 << prefix_bits
    width = 1.0 / segments
    centres = (np.arange(segments) + 0.5) * width

    # Exact surface, coloured by height. Kept opaque so the gradient reads; the
    # planes above it are translucent instead.
    grid = np.linspace(0.0, 1.0, 121)
    m_1, m_2 = np.meshgrid(grid, grid, indexing="xy")
    exact = m_1 * m_2
    surface = axis.plot_surface(
        m_1,
        m_2,
        exact,
        cmap=HEIGHT_CMAP,
        vmin=0.0,
        vmax=1.0,
        linewidth=0.0,
        antialiased=True,
        rstride=1,
        cstride=1,
        shade=False,
        alpha=0.92,
        zorder=1,
    )

    # The planes are drawn as outlines, not filled patches. At k=2 a plane and
    # the surface differ by at most 2^-6, so a filled patch is depth-sorted
    # arbitrarily against the surface and disappears under it; the edges of each
    # tile show the faceting without competing for the same pixels.
    for i, mu_i in enumerate(centres):
        for j, nu_j in enumerate(centres):
            lo_1, hi_1 = i * width, (i + 1) * width
            lo_2, hi_2 = j * width, (j + 1) * width
            corners_1 = [lo_1, hi_1, hi_1, lo_1, lo_1]
            corners_2 = [lo_2, lo_2, hi_2, hi_2, lo_2]
            heights = [
                nu_j * c1 + mu_i * c2 - mu_i * nu_j
                for c1, c2 in zip(corners_1, corners_2)
            ]
            axis.plot(
                corners_1,
                corners_2,
                heights,
                color=COLOR_TILE,
                linewidth=0.95,
                solid_capstyle="round",
                zorder=6,
            )

    ticks = np.array([0.0, 0.5, 1.0])
    axis.set(
        xlim=(0.0, 1.0),
        ylim=(0.0, 1.0),
        zlim=(0.0, 1.02),
        xlabel=r"$m_1$",
        ylabel=r"$m_2$",
        zlabel=r"$m_1m_2$",
    )
    axis.set_xticks(ticks)
    axis.set_yticks(ticks)
    axis.set_zticks(ticks)
    axis.tick_params(pad=4.0)
    axis.xaxis.labelpad = 1
    axis.yaxis.labelpad = 1
    axis.zaxis.labelpad = 1
    axis.view_init(elev=25, azim=-127)
    axis.set_box_aspect((1.0, 1.0, 0.70))
    axis.set_title(
        rf"(b) The mantissa product, tiled ($k={prefix_bits}$)",
        pad=7,
        fontweight="bold",
    )

    legend_handles = [
        Patch(facecolor=plt.get_cmap(HEIGHT_CMAP)(0.55), edgecolor="none",
              label=r"Exact $m_1m_2$"),
        Patch(
            facecolor=COLOR_TILE,
            alpha=0.45,
            edgecolor=COLOR_TILE,
            label=rf"${segments}\!\times\!{segments}$ prefix planes",
        ),
    ]
    axis.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(-0.02, 0.97),
        frameon=False,
        borderaxespad=0.0,
        handlelength=1.5,
    )
    return surface, 2.0 ** (-2 * prefix_bits - 2)


def render(output_stem: Path, prefix_bits: int) -> list[Path]:
    """Render the figure and return the three generated artifact paths."""

    if prefix_bits < 1 or prefix_bits > 4:
        raise ValueError("prefix_bits must be between 1 and 4 for a legible concept figure")

    configure_matplotlib()
    figure = plt.figure(figsize=(7.2, 3.05), constrained_layout=False)
    grid = figure.add_gridspec(1, 2, width_ratios=(1.0, 1.16), wspace=0.20)
    left = figure.add_subplot(grid[0, 0])
    right = figure.add_subplot(grid[0, 1], projection="3d")

    gelu_error = draw_gelu_panel(left)
    surface, residual_bound = draw_mantissa_panel(right, prefix_bits)
    figure.subplots_adjust(left=0.075, right=0.93, bottom=0.16, top=0.89)

    bar = figure.colorbar(surface, ax=right, shrink=0.52, aspect=13, pad=0.02)
    bar.set_label("height", fontsize=7.0, labelpad=2)
    bar.set_ticks([0.0, 0.5, 1.0])
    bar.ax.tick_params(labelsize=6.4, length=2.0)
    bar.outline.set_linewidth(0.6)

    output_stem = output_stem.with_suffix("")
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    outputs = [output_stem.with_suffix(suffix) for suffix in (".pdf", ".svg", ".png")]
    figure.savefig(outputs[0], bbox_inches="tight", pad_inches=0.025)
    figure.savefig(outputs[1], bbox_inches="tight", pad_inches=0.025)
    figure.savefig(outputs[2], dpi=600, bbox_inches="tight", pad_inches=0.025)
    plt.close(figure)

    print(f"four-segment GELU max |error|: {gelu_error:.4e}")
    print(f"k={prefix_bits} tile residual bound: {residual_bound:.4e}")
    for output in outputs:
        print(f"wrote {output}")
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Render the Introduction PLA concept figure.")
    parser.add_argument(
        "--prefix-bits",
        type=int,
        default=2,
        help="mantissa-prefix width for panel (b); 2 gives a legible 4x4 tiling",
    )
    parser.add_argument(
        "--output-stem",
        type=Path,
        default=Path(__file__).resolve().parent / "fig_pla_concept",
        help="output path without extension; PDF, SVG, and PNG are written",
    )
    args = parser.parse_args()
    render(args.output_stem, args.prefix_bits)


if __name__ == "__main__":
    main()
