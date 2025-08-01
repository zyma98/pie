use std::time::{Duration, Instant};

#[inferlet::main]
async fn main() -> Result<(), String> {
    let flag = inferlet::receive().await;

    let mut total_duration = Duration::ZERO;

    let PING_COUNT = 24;

    for i in 1..=PING_COUNT {
        let start = Instant::now();
        let resp = inferlet::debug_query("hello");
        let elapsed = start.elapsed();
        total_duration += elapsed;
    }

    let avg_latency = total_duration / PING_COUNT as u32;

    // if print_output {
    //
    // }

    inferlet::send(&avg_latency.as_micros().to_string());
    //println!("Average Latency: {:?}", avg_latency);

    Ok(())
}
