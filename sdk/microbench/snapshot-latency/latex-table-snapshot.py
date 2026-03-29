#!/usr/bin/env python3
"""Parse snapshot-latency benchmark output and generate a LaTeX table."""

import re
import sys

SUMMARY_ITEMS = [
    (re.compile(r"Avg first-call latency WITH snapshot:\s+([\d.]+) us"),
     r"\microbenchSnapshotFirstCallWith", "first call with snapshot"),
    (re.compile(r"Avg first-call latency WITHOUT snapshot:\s+([\d.]+) us"),
     r"\microbenchSnapshotFirstCallWithout", "first call w/o snapshot"),
    (re.compile(r"Avg snapshot creation latency:\s+([\d.]+) us"),
     r"\microbenchSnapshotCreation", "taking snapshot"),
]


def parse(lines: list[str]) -> dict[str, float]:
    """Return {var_name: latency_ms} from benchmark output summary."""
    results = {}
    for line in lines:
        line = line.strip()
        for pattern, var_name, _ in SUMMARY_ITEMS:
            m = pattern.search(line)
            if m:
                results[var_name] = float(m.group(1)) / 1000.0
    return results


def latex_table(results: dict[str, float]) -> str:
    lines = []
    for _, var_name, _ in SUMMARY_ITEMS:
        val = results.get(var_name, float("nan"))
        lines.append(rf"\newcommand{{{var_name}}}{{{val:.2f}}}")

    lines.append("")
    lines.append(r"\begin{table}[t]")
    lines.append(r"    \begin{tabular}{|l|r|}")
    lines.append(r"    \hline")
    lines.append(
        r"    Action                   "
        r"& Latency (ms)                         \\ \hline\hline"
    )
    for _, var_name, label in SUMMARY_ITEMS:
        lines.append(
            f"    {label:<25}& {var_name:<37}\\\\ \\hline"
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
