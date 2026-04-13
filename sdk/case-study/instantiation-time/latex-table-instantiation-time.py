#!/usr/bin/env python3
"""Parse instantiation-time benchmark output and generate a LaTeX table."""

import re
import sys

CASE_RE = re.compile(r"^\s+(\S+)\s+\(\d+ deps\):")
METRIC_RE = re.compile(
    r"^\s+(store\+linker|deps|app|total):\s+mean=([\d.]+)\s+min=([\d.]+)\s+max=([\d.]+)\s+stddev=([\d.]+)"
)

CASES = [
    ("text-completion", "text completion (mono)", "Mono", 0),
    ("text-completion-inferlib", "text completion", "TextComp", 1),
    ("constrained-decoding", "constrained decoding", "ConstrDec", 2),
    ("json-schema-validation", "json schema validation", "JsonSchema", 3),
    ("template-generation", "templated generation", "TemplateGen", 4),
]

LIBS = ["Inference", "Constraint", "Schema", "Template"]

# (result_key, variable_suffix) in table column order
METRICS = [
    ("store_linker", "StoreLinker"),
    ("deps", "Deps"),
    ("app", "App"),
    ("total", "Total"),
]

VAR_PREFIX = r"\caseInstTime"


METRIC_KEY_MAP = {
    "store+linker": "store_linker",
    "deps": "deps",
    "app": "app",
    "total": "total",
}


def parse(lines: list[str]) -> dict[str, dict[str, tuple[float, float]]]:
    """Return {case_name: {metric: (mean, stddev)}} from the Overall Summary.

    Values are in microseconds.
    """
    results: dict[str, dict[str, tuple[float, float]]] = {}
    in_summary = False
    current_case = None
    for line in lines:
        if "Overall Summary" in line:
            in_summary = True
            continue
        if not in_summary:
            continue
        m = CASE_RE.match(line)
        if m:
            current_case = m.group(1)
            results.setdefault(current_case, {})
            continue
        if current_case:
            m = METRIC_RE.match(line)
            if m:
                key = METRIC_KEY_MAP[m.group(1)]
                mean_val = float(m.group(2))
                stddev_val = float(m.group(5))
                results[current_case][key] = (mean_val, stddev_val)
                if key == "total":
                    current_case = None
    return results


def latex_table(results: dict[str, dict[str, tuple[float, float]]]) -> str:
    lines: list[str] = []

    for case_key, _, tag, _ in CASES:
        data = results.get(case_key, {})
        for result_key, metric_suffix in METRICS:
            mean_us, stddev_us = data.get(result_key, (float("nan"), float("nan")))
            mean_ms = mean_us / 1000.0
            stddev_ms = stddev_us / 1000.0
            var = f"{VAR_PREFIX}{tag}{metric_suffix}"
            var_sd = f"{VAR_PREFIX}{tag}{metric_suffix}Sd"
            lines.append(rf"\newcommand{{{var}}}{{{mean_ms:.2f}}}")
            lines.append(rf"\newcommand{{{var_sd}}}{{{stddev_ms:.2f}}}")

    lines.append("")
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\begin{tabular}{|l||cccc||rrrr|}")
    lines.append(r"\hline")

    lines.append(
        r"\multirow{2}{*}{Application Name} "
        r"& \multicolumn{4}{c||}{Library Name} "
        r"& \multicolumn{4}{c|}{Instantiation Latency (ms)} \\ \cline{2-9}"
    )

    lines.append(
        r"& \multicolumn{1}{c|}{Inference} "
        r"& \multicolumn{1}{c|}{Constraint} "
        r"& \multicolumn{1}{c|}{Schema} "
        r"& Template "
        r"& \multicolumn{1}{c|}{Store \& Linker} "
        r"& \multicolumn{1}{c|}{Dependency} "
        r"& \multicolumn{1}{c|}{App} "
        r"& \multicolumn{1}{c|}{Total} \\ \hline\hline"
    )

    for case_key, label, tag, num_deps in CASES:
        cols = [f"{label:<39}"]

        for i in range(len(LIBS)):
            check = r"\CheckmarkBold" if i < num_deps else ""
            if i < len(LIBS) - 1:
                cols.append(rf"\multicolumn{{1}}{{c|}}{{{check}}}")
            else:
                cols.append(check)

        for i, (_, metric_suffix) in enumerate(METRICS):
            var = f"{VAR_PREFIX}{tag}{metric_suffix}"
            var_sd = f"{VAR_PREFIX}{tag}{metric_suffix}Sd"
            cell = rf"{var}\,({var_sd})"
            if i < len(METRICS) - 1:
                cols.append(rf"\multicolumn{{1}}{{r|}}{{{cell}}}")
            else:
                cols.append(cell)

        lines.append(" & ".join(cols) + r" \\ \hline")

    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")

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
