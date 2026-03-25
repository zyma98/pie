use inferlib_run_bindings::{Args, Result};

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

    let result = inferlib_h2r_benchmark_bindings::run_benchmark(warmup, iterations);
    println!("{result}");

    Ok(())
}
