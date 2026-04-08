#!/usr/bin/env python3
"""Parse memory-consumption benchmark output and draw line plots.

Usage:
    line-plot-memory.py <result.txt> [output.pdf]
"""

import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

fm._load_fontmanager(try_read_cache=False)
plt.rcParams["font.family"] = "Linux Libertine O"
plt.rcParams["pdf.fonttype"] = 42

# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

CASE_HEADER_RE = re.compile(r"Case\s+\d+:.*\((.+)\)")
TREE_RSS_RE = re.compile(r"PID\s+\d+\s+RSS\s+(\d+)\s+KB")
TICK_RE = re.compile(r"submitted\s+(\d+)/\d+\s+RSS=(\d+)\s+KB")
IDLE_RE = re.compile(r"Idle engine RSS:\s+(\d+)\s+KB")


def parse(lines: list[str]) -> tuple[dict[str, list[tuple[int, int]]], int | None]:
    """Return (data, idle_rss_kb).

    data: {case_name: [(n_submitted, rss_kb), ...]}
    """
    data: dict[str, list[tuple[int, int]]] = {}
    idle_rss_kb = None
    current_case = None
    tree_rss = 0
    in_tree = False

    for line in lines:
        m = IDLE_RE.search(line)
        if m:
            idle_rss_kb = int(m.group(1))
            continue

        m = CASE_HEADER_RE.search(line)
        if m:
            current_case = m.group(1).strip()
            data[current_case] = []
            tree_rss = 0
            in_tree = False
            continue

        if current_case is None:
            continue

        if "Process tree" in line:
            in_tree = True
            tree_rss = 0
            continue

        if in_tree:
            m = TREE_RSS_RE.search(line)
            if m:
                tree_rss += int(m.group(1))
                continue
            else:
                if tree_rss > 0 and not data[current_case]:
                    data[current_case].append((0, tree_rss))
                in_tree = False

        m = TICK_RE.search(line)
        if m:
            data[current_case].append((int(m.group(1)), int(m.group(2))))

    return data, idle_rss_kb


# ---------------------------------------------------------------------------
# Style constants (matching bar-plot-cold-start.py)
# ---------------------------------------------------------------------------

COLORS = ["#7BAFD4", "#8FCA8F", "#E8D0A0", "#D9A8A8"]


def darken(hex_color: str, factor: float = 0.20) -> str:
    """Blend *hex_color* toward black by *factor* (0 = unchanged, 1 = black)."""
    r = int(hex_color[1:3], 16)
    g = int(hex_color[3:5], 16)
    b = int(hex_color[5:7], 16)
    r = int(r * (1 - factor))
    g = int(g * (1 - factor))
    b = int(b * (1 - factor))
    return f"#{r:02x}{g:02x}{b:02x}"


RUST_CASES = [
    ("rust monolithic", "Monolithic",          darken(COLORS[0]), "s", "-"),
    ("rust static",     "Static",  darken(COLORS[1]), "o", "-"),
    ("rust dynamic",    "Dynamic", darken(COLORS[3]), "X", "-"),
]

PYTHON_CASES = [
    ("python full",   "Self-contained", darken(COLORS[0]), "s", "-"),
    ("python shared", "Factored",       darken(COLORS[3]), "X", "-"),
]

BOLD_KEYS = {"rust dynamic", "python shared"}

ROW_TITLE_SPECS = [
    [("rust monolithic", "monolithic"),
     ("rust static",     "static"),
     ("rust dynamic",    "dynamic")],
    [("python full",   "self-contained"),
     ("python shared", "factored")],
]

ROW_TITLE_COLORS = [
    [COLORS[0], COLORS[1], COLORS[3]],
    [COLORS[0], COLORS[3]],
]


def _per_instance_mib(data, idle_rss_kb):
    """Return {case_key: per_instance_MiB}."""
    result = {}
    for key, pts in data.items():
        if not pts:
            continue
        max_n, max_rss = pts[-1]
        if max_n > 0 and idle_rss_kb is not None:
            result[key] = (max_rss - idle_rss_kb) / max_n / 1024
    return result


def _draw_row_titles(fig, axes, per_inst, y_sizes=1.12, fontsize=10):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    for i, ax in enumerate(axes):
        segments = ROW_TITLE_SPECS[i]
        seg_colors = ROW_TITLE_COLORS[i]

        x = 0.0
        for si, (case_key, short_label) in enumerate(segments):
            val = per_inst.get(case_key, 0)
            text = f"{short_label} {val:.1f} MiB"

            t = ax.text(x, y_sizes, text, transform=ax.transAxes,
                        fontsize=fontsize, va="bottom", ha="left",
                        bbox=dict(facecolor=seg_colors[si], edgecolor="none",
                                  alpha=0.45, pad=1.5))

            fig.canvas.draw()
            ext = t.get_window_extent(renderer)
            inv = ax.transAxes.inverted()
            x = inv.transform((ext.x1, ext.y0))[0] + 0.01


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_subplot(ax, data, cases, title, *, xlim=None, ylim=None, xtick_step=None,
                  show_xlabel=True):
    for key, label, color, marker, ls in cases:
        pts = data.get(key, [])
        if not pts:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] / 1_000_000 for p in pts]
        bold = key in BOLD_KEYS
        ms = 6.0 if marker == "X" else 5.0 if marker == "o" else 4.5
        ax.plot(
            xs, ys,
            color=color, marker=marker, markersize=ms,
            linewidth=2.2 if bold else 1.5,
            linestyle=ls, label=label,
            markeredgecolor="white", markeredgewidth=0.5,
        )

    if show_xlabel:
        ax.set_xlabel("Number of Application Instance", fontsize=10)
    ax.set_ylabel("Memory (GiB)", fontsize=10)
    ax.text(0, 1.35, title, transform=ax.transAxes,
            fontsize=10.5, fontweight="demibold", va="bottom", ha="left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linewidth=0.3, alpha=0.5)

    ax.set_xlim(xlim if xlim else (0, None))
    ax.set_ylim(ylim if ylim else (0, None))
    if ylim:
        import numpy as np
        ax.set_yticks(np.arange(ylim[0], ylim[1] + 1, 20))
    if xtick_step and xlim:
        import numpy as np
        ax.set_xticks(np.arange(xlim[0], xlim[1] + 1, xtick_step))


def _add_legend(ax, cases, legend_y=1.4):
    """Add legend to an axes. Call after tight_layout so it doesn't affect sizing."""
    leg = ax.legend(
        fontsize=9.5, framealpha=0.9,
        loc="lower center", bbox_to_anchor=(0.5, legend_y),
        ncol=len(cases), borderaxespad=0,
    )
    for text in leg.get_texts():
        for key, label, *_ in cases:
            if text.get_text() == label and key in BOLD_KEYS:
                text.set_fontweight("bold")


def plot(data, idle_rss_kb, out_path):
    fig, axes = plt.subplots(2, 1, figsize=(4.2, 3.6))

    _plot_subplot(axes[0], data, RUST_CASES, "Rust — Templated Generation",
                  xlim=(0, 2000), ylim=(0, 80), xtick_step=250, show_xlabel=False)
    _plot_subplot(axes[1], data, PYTHON_CASES, "Python — Watermarking (Using NumPy)",
                  xlim=(0, 720), ylim=(0, 80), xtick_step=90)

    fig.tight_layout(h_pad=1.5)

    _add_legend(axes[0], RUST_CASES, legend_y=1.65)
    _add_legend(axes[1], PYTHON_CASES, legend_y=1.65)

    per_inst = _per_instance_mib(data, idle_rss_kb)
    _draw_row_titles(fig, axes, per_inst, y_sizes=1.10, fontsize=10.5)

    fig.savefig(out_path, bbox_inches="tight")
    print(f"Saved plot to {out_path}")


# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <result.txt> [output.pdf]", file=sys.stderr)
        sys.exit(1)

    with open(sys.argv[1]) as f:
        data, idle_rss_kb = parse(f.readlines())

    out_path = (
        Path(sys.argv[2]) if len(sys.argv) >= 3
        else Path(__file__).resolve().parent / "memory-consumption.pdf"
    )
    plot(data, idle_rss_kb, out_path)


if __name__ == "__main__":
    main()
