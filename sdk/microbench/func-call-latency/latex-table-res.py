#!/usr/bin/env python3
"""Parse func-call-latency benchmark output and generate a LaTeX table for resecho results."""

import re
import sys

SECTION_RE = re.compile(r"^=== (.+) ===$")
LATENCY_RE = re.compile(r"Per-call latency:\s+([\d.]+) ns")

RESOURCE_COUNTS = [1, 2, 4, 8]
NUM_WORDS = {1: "One", 2: "Two", 4: "Four", 8: "Eight"}

VAR_PREFIX = r"\microbenchFuncLatency"


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
    for r in RESOURCE_COUNTS:
        dyn_key = f"R2R ResEcho Dynamic Composition Resources={r}"
        sta_key = f"R2R ResEcho Static Composition Resources={r}"
        dyn_us = results.get(dyn_key, float("nan")) / 1000.0
        sta_us = results.get(sta_key, float("nan")) / 1000.0
        w = NUM_WORDS[r]
        var_defs.append((f"{VAR_PREFIX}DynamicRes{w}", dyn_us))
        var_defs.append((f"{VAR_PREFIX}StaticRes{w}", sta_us))

    lines = []
    for var, val in var_defs:
        lines.append(rf"\newcommand{{{var}}}{{{val:.2f}}}")

    lines.append("")
    lines.append(r"\begin{table}[t]")
    lines.append(r"    \begin{tabular}{|l|r|r|}")
    lines.append(r"    \hline")
    lines.append(
        r"    \#Resources           "
        r"& Composition & Latency (\textmu s)               \\ \hline\hline"
    )
    cline = r"\cline{2-3}"
    for r in RESOURCE_COUNTS:
        w = NUM_WORDS[r]
        dyn_var = f"{VAR_PREFIX}DynamicRes{w}"
        sta_var = f"{VAR_PREFIX}StaticRes{w}"
        row_cmd = rf"\multirow{{2}}{{*}}{{{r}}}"
        lines.append(
            f"    {row_cmd:<22}& {'dynamic':<8}& {dyn_var:<39}\\\\"
        )
        lines.append(
            f"    {cline:<22}& {'static':<8}& {sta_var:<39}\\\\ \\hline"
        )
    lines.append(r"    \end{tabular}")
    lines.append(r"\end{table}")

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
