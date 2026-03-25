#!/usr/bin/env python3
"""Parse func-call-latency benchmark output and generate LaTeX variables for H2R, R2H, and monolithic results."""

import re
import sys

SECTION_RE = re.compile(r"^=== (.+) ===$")
LATENCY_RE = re.compile(r"Per-call latency:\s+([\d.]+) ns")

VARS = [
    ("Running guest-to-host benchmark (R2H)", r"\microbench_func_latency_r2h"),
    ("Running host-to-guest benchmark (H2R)", r"\microbench_func_latency_h2r"),
    ("Running monolithic intra-component benchmark", r"\microbench_func_latency_mono"),
]


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


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <result_file>", file=sys.stderr)
        sys.exit(1)
    with open(sys.argv[1]) as f:
        results = parse(f.readlines())
    for section_key, var_name in VARS:
        val_us = results.get(section_key, float("nan")) / 1000.0
        print(rf"\newcommand{{{var_name}}}{{{val_us:.2f}}}")


if __name__ == "__main__":
    main()
