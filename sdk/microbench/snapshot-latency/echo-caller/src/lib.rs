use inferlib_run_bindings::{Args, Result};

#[inferlib_macros::main]
async fn main(mut args: Args) -> Result<()> {
    let iterations: u64 = args
        .value_from_str(["-n", "--iterations"])
        .unwrap_or(1);
    let warmup: u64 = args
        .value_from_str(["-w", "--warmup"])
        .unwrap_or(0);

    let result = snapshot_benchmark_bindings::run_benchmark(warmup, iterations);
    println!("{result}");

    Ok(())
}
