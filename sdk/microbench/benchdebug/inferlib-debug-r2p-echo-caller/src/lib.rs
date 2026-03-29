use inferlib_run_bindings::{Args, Result};
use std::hint::black_box;
use std::time::Instant;

const WARMUP_ITERATIONS: u64 = 100_000;
const BENCH_ITERATIONS: u64 = 10_000_000;

#[inferlib_macros::main]
async fn main(mut args: Args) -> Result<()> {
    let iterations: u64 = args
        .value_from_str(["-n", "--iterations"])
        .unwrap_or(BENCH_ITERATIONS);
    let warmup: u64 = args
        .value_from_str(["-w", "--warmup"])
        .unwrap_or(WARMUP_ITERATIONS);

    for _ in 0..warmup {
        black_box(inferlib_debug_r2r_echo_callee_bindings::echo("hello"));
    }

    let start = Instant::now();
    for _ in 0..iterations {
        black_box(inferlib_debug_r2r_echo_callee_bindings::echo("hello"));
    }
    let elapsed = start.elapsed();

    let total_ns = elapsed.as_nanos() as f64;
    let per_call_ns = total_ns / iterations as f64;

    println!("Cross-component echo() call benchmark (R2P)");
    println!("  Warmup iterations:  {warmup}");
    println!("  Bench iterations:   {iterations}");
    println!("  Total elapsed:      {elapsed:?}");
    println!("  Per-call latency:   {per_call_ns:.1} ns");

    Ok(())
}
