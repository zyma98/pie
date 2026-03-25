"""
Microbenchmark: cross-component echo() call latency (Python-to-Rust).

Measures the per-call overhead of invoking an echo function exported by
a Rust callee component from a Python caller component, going through
the dynamic linking layer.
"""

import time

from echo_callee_bindings import echo
from inference_bindings import set_return
from run_bindings import get_arguments

WARMUP_ITERATIONS = 100_000
BENCH_ITERATIONS = 10_000_000


def main() -> None:
    args = get_arguments()
    iterations = int(args.get("n", args.get("iterations", BENCH_ITERATIONS)))
    warmup = int(args.get("w", args.get("warmup", WARMUP_ITERATIONS)))

    for _ in range(warmup):
        echo("hello")

    start = time.perf_counter_ns()
    for _ in range(iterations):
        echo("hello")
    elapsed_ns = time.perf_counter_ns() - start

    per_call_ns = elapsed_ns / iterations
    elapsed_s = elapsed_ns / 1e9

    print(f"Cross-component echo() call benchmark (P2R)")
    print(f"  Warmup iterations:  {warmup}")
    print(f"  Bench iterations:   {iterations}")
    print(f"  Total elapsed:      {elapsed_s:.6f}s")
    print(f"  Per-call latency:   {per_call_ns:.1f} ns")

    set_return(f"Per-call latency: {per_call_ns:.1f} ns")


if __name__ == "__main__":
    main()
