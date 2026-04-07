#!/usr/bin/env python3
"""Parse cold-start-latency benchmark output and draw bar plots.

Usage:
    bar-plot-cold-start.py <result.txt> [output.pdf]
"""

import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.font_manager as fm
import numpy as np

fm._load_fontmanager(try_read_cache=False)
plt.rcParams["font.family"] = "Linux Libertine O"
plt.rcParams["pdf.fonttype"] = 42

# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

BW_HEADER_RE = re.compile(r"---\s+(\w+)\s+\(~\d+\s+Mbps\)\s+---")
RESULT_RE = re.compile(
    r"^\s+(\S+):\s+upload=([\d.]+)\s+s\s+compile=([\d.]+)\s+s\s+total=([\d.]+)\s+s"
)


SIZE_RE = re.compile(r"^\s+(\S+):\s+([\d.]+)\s+MB")


def parse(lines: list[str]) -> tuple[
    dict[tuple[str, str], dict[str, float]],
    dict[str, float],
]:
    """Return (results, sizes).

    results: {(bw_name, case_name): {"upload": f, "compile": f, "total": f}}
    sizes:   {case_name: size_in_MB}
    """
    results: dict[tuple[str, str], dict[str, float]] = {}
    sizes: dict[str, float] = {}
    current_bw = None
    in_summary = False
    in_sizes = False
    for line in lines:
        if "Wasm file sizes" in line:
            in_sizes = True
            continue
        if in_sizes:
            m = SIZE_RE.match(line)
            if m:
                sizes[m.group(1)] = float(m.group(2))
                continue
            else:
                in_sizes = False
        if "Overall Summary" in line:
            in_summary = True
            continue
        if not in_summary:
            continue
        m = BW_HEADER_RE.search(line)
        if m:
            current_bw = m.group(1)
            continue
        if current_bw:
            m = RESULT_RE.match(line)
            if m:
                results[(current_bw, m.group(1))] = {
                    "upload": float(m.group(2)),
                    "compile": float(m.group(3)),
                    "total": float(m.group(4)),
                }
    return results, sizes


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BANDWIDTHS = [("slow", "10 Mbps"), ("medium", "100 Mbps"), ("fast", "1000 Mbps")]

ROW1_CASES = [
    ("rust-mono-template-generation", "Monolithic"),
    ("rust-static-template-generation", "Static Composition"),
    ("rust-dynamic-template-generation", "Dynamic Composition"),
]

ROW23_CASES = [
    ("python-full-dynamic-template-generation", "Self-contained"),
    ("python-factored-dynamic-template-generation", "Factored"),
]

ROW23_CASES_WM = [
    ("python-full-dynamic-watermarking-numpy", "Self-contained"),
    ("python-factored-dynamic-watermarking-numpy", "Factored"),
]

# Blue, Green, Yellow, Red
COLORS = ["#7BAFD4", "#8FCA8F", "#E8D0A0", "#D9A8A8"]
HATCHES = ["..", "///", "\\\\\\", ""]

ROW1_COLORS = [COLORS[0], COLORS[1], COLORS[3]]
ROW1_HATCHES = [HATCHES[1], HATCHES[2], HATCHES[0]]

ROW23_COLORS = [COLORS[0], COLORS[3]]
ROW23_HATCHES = [HATCHES[1], HATCHES[0]]

ROW_TITLE_SPECS = [
    ("Rust — Templated Generation", [
        ("rust-mono-template-generation", "monolithic"),
        ("rust-static-template-generation", "static"),
        ("rust-dynamic-template-generation", "dynamic"),
    ]),
    ("Python+Rust — Templated Generation", [
        ("python-full-dynamic-template-generation", "self-contained"),
        ("python-factored-dynamic-template-generation", "factored"),
    ]),
    ("Python+Rust — Watermarking (Using NumPy)", [
        ("python-full-dynamic-watermarking-numpy", "self-contained"),
        ("python-factored-dynamic-watermarking-numpy", "factored"),
    ]),
]

ROW_TITLE_COLORS = [ROW1_COLORS, ROW23_COLORS, ROW23_COLORS]


def _draw_row_titles(fig, axes, sizes,
                     y_heading_row0, y_sizes_row0,
                     y_heading_row12, y_sizes_row12,
                     fontsize=10):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    for i, ax in enumerate(axes):
        heading, segments = ROW_TITLE_SPECS[i]
        seg_colors = ROW_TITLE_COLORS[i]

        y_heading = y_heading_row0 if i == 0 else y_heading_row12
        y_sizes = y_sizes_row0 if i == 0 else y_sizes_row12

        ax.text(0, y_heading, heading, transform=ax.transAxes,
                fontsize=fontsize, va="bottom", ha="left",
                fontweight="demibold")

        x = 0.0
        for si, (case_key, short_label) in enumerate(segments):
            size_val = sizes.get(case_key, 0)
            text = f"{short_label} {size_val:.2f} MiB"

            t = ax.text(x, y_sizes, text, transform=ax.transAxes,
                        fontsize=fontsize, va="bottom", ha="left",
                        bbox=dict(facecolor=seg_colors[si], edgecolor="none",
                                  alpha=0.45, pad=1.5))

            fig.canvas.draw()
            ext = t.get_window_extent(renderer)
            inv = ax.transAxes.inverted()
            x = inv.transform((ext.x1, ext.y0))[0] + 0.01

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------


def darken(hex_color: str, factor: float = 0.25) -> str:
    """Blend *hex_color* toward black by *factor* (0 = unchanged, 1 = black)."""
    r = int(hex_color[1:3], 16)
    g = int(hex_color[3:5], 16)
    b = int(hex_color[5:7], 16)
    r = int(r * (1 - factor))
    g = int(g * (1 - factor))
    b = int(b * (1 - factor))
    return f"#{r:02x}{g:02x}{b:02x}"


BOLD_CASES = {
    "rust-dynamic-template-generation",
    "python-factored-dynamic-template-generation",
    "python-factored-dynamic-watermarking-numpy",
}

MAX_VARIANTS = max(len(ROW1_CASES), len(ROW23_CASES), len(ROW23_CASES_WM))
BAR_WIDTH = 0.22
GROUP_GAP = 0.15


def _draw_break_mark(ax, x, y_top, bar_w, y_cap):
    """Draw a zigzag across a clipped bar to indicate truncation."""
    hw = bar_w / 2
    dy = y_cap * 0.025
    n_pts = 9
    xs = np.linspace(x - hw, x + hw, n_pts)
    ys = [y_top + (dy if i % 2 == 0 else -dy) for i in range(n_pts)]
    ax.plot(xs, ys, color="white", linewidth=2.5, zorder=5, clip_on=False)
    ax.plot(xs, ys, color="black", linewidth=0.6, zorder=6, clip_on=False)


def plot_row(
    ax: plt.Axes,
    results: dict,
    cases: list[tuple[str, str]],
    colors: list[str],
    hatches: list[str],
    scale_bws: list[str] | None = None,
) -> None:
    n_groups = len(BANDWIDTHS)
    n_variants = len(cases)

    x_centres = np.arange(n_groups) * (MAX_VARIANTS * BAR_WIDTH + GROUP_GAP)

    all_totals: list[list[float]] = []
    all_uploads: list[list[float]] = []
    all_compiles: list[list[float]] = []

    for vi, (case_key, case_label) in enumerate(cases):
        uploads = [
            results.get((bw_key, case_key), {}).get("upload", 0)
            for bw_key, _ in BANDWIDTHS
        ]
        compiles = [
            results.get((bw_key, case_key), {}).get("compile", 0)
            for bw_key, _ in BANDWIDTHS
        ]
        totals = [u + c for u, c in zip(uploads, compiles)]
        all_totals.append(totals)
        all_uploads.append(uploads)
        all_compiles.append(compiles)

    if scale_bws is not None:
        scale_indices = [
            j for j, (bw_key, _) in enumerate(BANDWIDTHS) if bw_key in scale_bws
        ]
        ref_max = max(
            all_totals[vi][j] for vi in range(n_variants) for j in scale_indices
        ) or 1.0
    else:
        ref_max = max(h for hs in all_totals for h in hs) or 1.0

    y_cap = ref_max * 1.14

    for vi, (case_key, case_label) in enumerate(cases):
        offsets = x_centres + (vi - (n_variants - 1) / 2) * BAR_WIDTH
        for j in range(n_groups):
            u = all_uploads[vi][j]
            c = all_compiles[vi][j]
            total = u + c
            clipped = total > y_cap

            draw_u = min(u, y_cap)
            draw_c = min(c, max(y_cap - u, 0)) if not clipped else max(y_cap - draw_u, 0)

            ax.bar(
                offsets[j], draw_u, BAR_WIDTH,
                color=darken(colors[vi]), edgecolor="dimgray", linewidth=0.3,
                hatch=hatches[vi],
            )
            ax.bar(
                offsets[j], draw_c, BAR_WIDTH, bottom=draw_u,
                color=colors[vi], edgecolor="dimgray", linewidth=0.3,
                hatch=hatches[vi],
            )

            if clipped:
                _draw_break_mark(ax, offsets[j], y_cap, BAR_WIDTH, y_cap)

    text_offset = y_cap * 0.02

    for vi, (case_key, _) in enumerate(cases):
        offsets = x_centres + (vi - (n_variants - 1) / 2) * BAR_WIDTH
        bold = case_key in BOLD_CASES
        for j, (bw_key, _) in enumerate(BANDWIDTHS):
            h = all_totals[vi][j]
            if h > 0:
                label_y = min(h, y_cap) + text_offset
                ax.text(
                    offsets[j], label_y,
                    f"{h:.1f}s", ha="center", va="bottom", fontsize=10,
                    fontweight="bold" if bold else "normal",
                )

    left_edge = x_centres[0] - (MAX_VARIANTS / 2) * BAR_WIDTH - 0.05
    right_edge = x_centres[-1] + (MAX_VARIANTS / 2) * BAR_WIDTH + 0.05
    ax.set_xlim(left_edge, right_edge)
    ax.set_ylim(0, y_cap)
    ax.set_xticks(x_centres)
    ax.set_xticklabels([label for _, label in BANDWIDTHS], fontsize=11)
    ax.set_ylabel("Latency (s)", fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot(results: dict, sizes: dict[str, float], out_path: Path) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(4.5, 4.6))

    plot_row(axes[0], results, ROW1_CASES, ROW1_COLORS, ROW1_HATCHES)
    plot_row(axes[1], results, ROW23_CASES, ROW23_COLORS, ROW23_HATCHES,
             scale_bws=["medium", "fast"])
    plot_row(axes[2], results, ROW23_CASES_WM, ROW23_COLORS, ROW23_HATCHES,
             scale_bws=["medium", "fast"])

    axes[2].set_xlabel("Network Uplink Bandwidth", fontsize=11)

    def _build_legend(ax, cases, colors, hatches, columnspacing=2.0, legend_y=2.4):
        col0 = []
        bold_indices = set()
        for vi, (case_key, label) in enumerate(cases):
            if case_key in BOLD_CASES:
                bold_indices.add(vi)
            col0.append(mpatches.Patch(
                facecolor=colors[vi], edgecolor="dimgray", linewidth=0.3,
                hatch=hatches[vi], label=label,
            ))
        col1: list[mpatches.Patch] = [
            mpatches.Patch(
                facecolor="#cccccc", edgecolor="dimgray", linewidth=0.3,
                label="Compile + Instantiate",
            ),
            mpatches.Patch(
                facecolor="#666666", edgecolor="dimgray", linewidth=0.3,
                label="Binary Upload",
            ),
        ]
        while len(col1) < len(cases):
            col1.append(mpatches.Patch(
                facecolor="none", edgecolor="none", label="",
            ))
        handles = col0 + col1
        leg = ax.legend(
            handles=handles,
            loc="lower center", bbox_to_anchor=(0.45, legend_y),
            fontsize=10, ncol=2, framealpha=0.9,
            columnspacing=columnspacing, labelspacing=0.3,
        )
        for vi in bold_indices:
            leg.get_texts()[vi].set_fontweight("bold")

    fig.align_ylabels(axes)
    fig.tight_layout(h_pad=7.0)

    pos2 = axes[2].get_position()
    axes[2].set_position([pos2.x0, pos2.y0 + 0.13, pos2.width, pos2.height])

    _build_legend(axes[0], ROW1_CASES, ROW1_COLORS, ROW1_HATCHES,
                  columnspacing=-1.75, legend_y=2.25)
    _build_legend(axes[1], ROW23_CASES, ROW23_COLORS, ROW23_HATCHES,
                  legend_y=2.35)

    _draw_row_titles(fig, axes, sizes,
                     y_heading_row0=1.85, y_sizes_row0=1.40,
                     y_heading_row12=1.95, y_sizes_row12=1.5,
                     fontsize=11)
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Saved plot to {out_path}")


def main() -> None:
    if len(sys.argv) < 2:
        print(
            f"Usage: {sys.argv[0]} <result.txt> [output.pdf]",
            file=sys.stderr,
        )
        sys.exit(1)

    with open(sys.argv[1]) as f:
        results, sizes = parse(f.readlines())

    out_path = Path(sys.argv[2]) if len(sys.argv) >= 3 else (
        Path(__file__).resolve().parent / "cold-start-latency.pdf"
    )
    plot(results, sizes, out_path)


if __name__ == "__main__":
    main()
