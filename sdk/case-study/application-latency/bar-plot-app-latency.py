#!/usr/bin/env python3
"""Parse application-latency benchmark output and draw a normalised bar plot.

Usage:
    bar-plot-app-latency.py <result1> <result2> <result3> [output.pdf]

Each result file corresponds to one model. The plot has one shared legend row
on top, followed by one bar-plot row per model.
"""

import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

fm._load_fontmanager(try_read_cache=False)
plt.rcParams["font.family"] = "Linux Libertine O"
plt.rcParams["pdf.fonttype"] = 42

# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

SUMMARY_RE = re.compile(
    r"^\s+(\S+)\s+avg=\s*([\d.]+)\s*ms"
)

def parse(lines: list[str]) -> dict[str, float]:
    """Return {case_name: avg_ms} from the Summary section."""
    results: dict[str, float] = {}
    in_summary = False
    for line in lines:
        if "Summary" in line and "measured runs" in line:
            in_summary = True
            continue
        if not in_summary:
            continue
        m = SUMMARY_RE.match(line)
        if m:
            results[m.group(1)] = float(m.group(2))
    return results

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

GROUPS = [
    ("text-completion", "Text\nCompletion*"),
    ("prefix-tree", "Prefix\nTree*"),
    ("rot", "Recursion\nof Thought*"),
    ("cacheback", "Cacheback\nDecoding\u2020"),
    ("codeact", "Agent\nCodeAct\u2020"),
    ("template-gen", "Templated\nGeneration\u2021"),
]

VARIANTS = [
    ("monolithic", "Rust Monolithic"),
    ("rust-inferlib-static", "Rust Static Composition"),
    ("rust-inferlib", "Rust Dynamic Composition"),
    ("python-inferlib", "Python+Rust Dynamic Composition"),
]

COLORS = ["#7BAFD4", "#8FCA8F", "#E8D0A0", "#D9A8A8"]
HATCHES = ["..", "///", "\\\\\\", ""]

MODEL_LABELS = ["Qwen3-8B", "Qwen3-14B", "Qwen3-32B"]

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_row(ax: plt.Axes, results: dict[str, float], show_legend: bool) -> None:
    """Draw one row of the bar plot on the given axes."""
    n_groups = len(GROUPS)
    n_variants = len(VARIANTS)
    bar_width = 0.22
    group_gap = 0.18

    x_centres = np.arange(n_groups) * (n_variants * bar_width + group_gap)

    for vi, (variant_key, variant_label) in enumerate(VARIANTS):
        offsets = x_centres + (vi - (n_variants - 1) / 2) * bar_width
        heights = []
        raw_avgs = []
        for group_key, _ in GROUPS:
            mono_key = f"{group_key}/monolithic"
            var_key = f"{group_key}/{variant_key}"
            mono_val = results.get(mono_key)
            var_val = results.get(var_key)
            if mono_val and var_val:
                heights.append(var_val / mono_val * 100)
                raw_avgs.append(var_val / 1000)
            else:
                heights.append(0)
                raw_avgs.append(0)

        bars = ax.bar(
            offsets, heights, bar_width,
            label=variant_label if show_legend else None,
            color=COLORS[vi], edgecolor="dimgray", linewidth=0.3,
            hatch=HATCHES[vi],
        )

        is_python = variant_key == "python-inferlib"
        for bar, h, raw in zip(bars, heights, raw_avgs):
            if h > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    f"{raw:.1f}s", ha="center", va="bottom", fontsize=9,
                    fontweight="bold" if is_python else "normal",
                )

    ax.axhline(100, color="grey", linewidth=0.7, linestyle="--", zorder=0)

    left_edge = x_centres[0] - (n_variants / 2) * bar_width - 0.1
    right_edge = x_centres[-1] + (n_variants / 2) * bar_width + 0.1
    ax.set_xlim(left_edge, right_edge)
    ax.set_xticks(x_centres)
    ax.set_xticklabels([label for _, label in GROUPS], fontsize=10)
    ax.set_ylabel("Normalised\nLatency (%)", fontsize=10)
    ax.set_ylim(0, 110)
    ax.set_yticks([0, 50, 100])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot(all_results: list[dict[str, float]], out_path: Path) -> None:
    n_models = len(all_results)
    fig, axes = plt.subplots(
        n_models, 1, figsize=(9, 1.3 * n_models), sharex=True,
    )
    if n_models == 1:
        axes = [axes]

    for i, (ax, results) in enumerate(zip(axes, all_results)):
        plot_row(ax, results, show_legend=(i == 0))
        ax.set_title(MODEL_LABELS[i], fontsize=10, loc="left", pad=9)
        if i < n_models - 1:
            ax.set_xticklabels([])
            ax.set_xlabel("")

    leg = axes[0].legend(
        loc="lower center", bbox_to_anchor=(0.49, 1.39),
        fontsize=9, ncol=len(VARIANTS), framealpha=0.9,
    )
    for txt in leg.get_texts():
        if txt.get_text() == "Python+Rust Dynamic Composition":
            txt.set_fontweight("bold")

    fig.tight_layout(h_pad=0.75)
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Saved plot to {out_path}")


def main() -> None:
    if len(sys.argv) < 4:
        print(
            f"Usage: {sys.argv[0]} <result1> <result2> <result3> [output.pdf]",
            file=sys.stderr,
        )
        sys.exit(1)

    all_results = []
    for path in sys.argv[1:4]:
        with open(path) as f:
            all_results.append(parse(f.readlines()))

    out_path = Path(sys.argv[4]) if len(sys.argv) >= 5 else (
        Path(__file__).resolve().parent / "app-latency.pdf"
    )
    plot(all_results, out_path)


if __name__ == "__main__":
    main()
