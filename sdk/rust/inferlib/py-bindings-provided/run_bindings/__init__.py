"""
run_bindings - Python equivalent of inferlib run-bindings.

Provides argument parsing for inferlet applications,
analogous to pico-args::Arguments in Rust.
"""

from inference_bindings import get_arguments as _get_raw_arguments


def parse_args(raw_args: list[str]) -> dict[str, str | bool]:
    """Parse POSIX-style CLI arguments into a dict."""
    parsed: dict[str, str | bool] = {}
    i = 0
    while i < len(raw_args):
        arg = raw_args[i]
        if arg.startswith("-"):
            key = arg.lstrip("-")
            next_arg = raw_args[i + 1] if i + 1 < len(raw_args) else None
            if next_arg and not next_arg.startswith("-"):
                parsed[key] = next_arg
                i += 2
            else:
                parsed[key] = True
                i += 1
        else:
            i += 1
    return parsed


def get_arguments() -> dict[str, str | bool]:
    """Retrieve and parse CLI arguments for the inferlet."""
    return parse_args(_get_raw_arguments())
