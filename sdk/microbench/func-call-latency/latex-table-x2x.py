#!/usr/bin/env python3
"""Parse func-call-latency benchmark output and generate a LaTeX table."""

import re
import sys

SECTION_RE = re.compile(r"^=== (.+) ===$")
LATENCY_RE = re.compile(r"Per-call latency:\s+([\d.]+) ns")

ECHO_ROWS = [
    ("R2R Echo", "Rust to Rust", "r2r"),
    ("R2P Echo", "Rust to Python", "r2p"),
    ("P2R Echo", "Python to Rust", "p2r"),
    ("P2P Echo", "Python to Python", "p2p"),
]

VAR_PREFIX = r"\microbench_func_latency"


def parse(lines: list[str]) -> dict[str, float]:
    """Return {section_name: latency_ns} from benchmark output lines."""
    results = {}
    current_section = None
    for line in lines:
        line = line.strip()
        m = SECTION_RE.match(line)
        if m:
            current_section = m.group(1)
            continue
        if current_section:
            m = LATENCY_RE.search(line)
            if m:
                results[current_section] = float(m.group(1))
    return results


def latex_table(results: dict[str, float]) -> str:
    var_defs: list[tuple[str, float]] = []
    lines = [
        r"% Please add the following required packages to your document preamble:",
        r"% \usepackage{multirow}",
        r"\begin{table}[]",
        r"\begin{tabular}{|l|r|r|}",
        r"\hline",
        r"Caller and Callee                 "
        r"& \begin{tabular}[c]{@{}r@{}}Composition\\ Type\end{tabular} "
        r"& \begin{tabular}[c]{@{}r@{}}Latency\\ (us)\end{tabular} \\ \hline",
    ]
    for i, (key_prefix, label, tag) in enumerate(ECHO_ROWS):
        dyn_key = f"{key_prefix} Dynamic Composition"
        sta_key = f"{key_prefix} Static Composition"
        dyn_us = results.get(dyn_key, float("nan")) / 1000.0
        sta_us = results.get(sta_key, float("nan")) / 1000.0
        dyn_var = f"{VAR_PREFIX}_{tag}_dynamic"
        sta_var = f"{VAR_PREFIX}_{tag}_static"
        var_defs.append((dyn_var, dyn_us))
        var_defs.append((sta_var, sta_us))
        row_cmd = rf"\multirow{{2}}{{*}}{{{label}}}"
        lines.append(
            f"{row_cmd:<34}& {'dynamic':<73}& {dyn_var:<69} \\\\ \\cline{{2-3}} "
        )
        lines.append(
            f"{'':34}& {'static':<73}& {sta_var:<69} \\\\ \\hline"
        )
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    lines.append("")
    for var, val in var_defs:
        lines.append(rf"\newcommand{{{var}}}{{{val:.2f}}}")

    return "\n".join(lines)


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <result_file>", file=sys.stderr)
        sys.exit(1)
    with open(sys.argv[1]) as f:
        results = parse(f.readlines())
    print(latex_table(results))


if __name__ == "__main__":
    main()
