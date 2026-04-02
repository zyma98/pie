#!/usr/bin/env python3
"""Parse instantiation-time benchmark output and generate a LaTeX table."""

import re
import sys

CASE_RE = re.compile(r"^\s+(\S+)\s+\(\d+ deps\):")
DATA_RE = re.compile(
    r"store\+linker=([\d.]+)\s+deps=([\d.]+)\s+app=([\d.]+)\s+total=([\d.]+)\s+us"
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


def parse(lines: list[str]) -> dict[str, dict[str, float]]:
    """Return {case_name: {store_linker, deps, app, total}} from the Overall Summary."""
    results = {}
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
            continue
        if current_case:
            m = DATA_RE.search(line)
            if m:
                results[current_case] = {
                    "store_linker": float(m.group(1)),
                    "deps": float(m.group(2)),
                    "app": float(m.group(3)),
                    "total": float(m.group(4)),
                }
                current_case = None
    return results


def latex_table(results: dict[str, dict[str, float]]) -> str:
    lines: list[str] = []

    for case_key, _, tag, _ in CASES:
        data = results.get(case_key, {})
        for result_key, metric_suffix in METRICS:
            var = f"{VAR_PREFIX}{tag}{metric_suffix}"
            val_ms = data.get(result_key, float("nan")) / 1000.0
            lines.append(rf"\newcommand{{{var}}}{{{val_ms:.2f}}}")

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
            if i < len(METRICS) - 1:
                cols.append(rf"\multicolumn{{1}}{{r|}}{{{var}}}")
            else:
                cols.append(var)

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
