//! Demonstrates parallel text generation from forked contexts.
//!
//! This example creates a shared system prompt context, then forks it into
//! two independent contexts that generate responses concurrently. Both
//! generations share the KV cache from the common prefix.

use futures::future;
use inferlib_inference_bindings::{Context, GenerateFuture, Model, SamplerConfig, StopConfig};
use inferlib_run_bindings::{Args, Result};
use std::time::Instant;
use wstd::runtime::AsyncPollable;

const HELP: &str = "\
Usage: parallel-generation [OPTIONS]

A program to demonstrate parallel text generation from forked contexts.

Options:
  -n, --max-tokens <TOKENS>  Max tokens to generate for each prompt [default: 128]
  -h, --help                 Prints this help message";

fn make_stop_config(max_tokens: u32, eos: &[Vec<u32>]) -> StopConfig {
    StopConfig {
        max_tokens,
        eos_sequences: eos.to_vec(),
    }
}

async fn poll_generate(future: &GenerateFuture) -> String {
    loop {
        let pollable = future.pollable();
        AsyncPollable::new(pollable).wait_for().await;
        if let Some(result) = future.get() {
            return result;
        }
    }
}

async fn poll_flush(ctx: &Context) {
    if let Some(future) = ctx.flush_async() {
        let pollable = future.pollable();
        AsyncPollable::new(pollable).wait_for().await;
    }
}

#[inferlib_macros::main]
async fn main(mut args: Args) -> Result<()> {
    if args.contains(["-h", "--help"]) {
        println!("{}", HELP);
        return Ok(());
    }

    let max_num_outputs: usize = args.value_from_str(["-n", "--max-tokens"]).unwrap_or(128);

    let start = Instant::now();

    let model = Model::get_auto();
    let eos_tokens = model.eos_tokens();
    let common = Context::new(&model);

    common.fill_system("You are a helpful, respectful and honest assistant.");
    poll_flush(&common).await;

    let ctx1 = common.fork();
    let eos_tokens1 = eos_tokens.clone();
    let handle1 = async {
        ctx1.fill_user("Explain Pulmonary Embolism");

        let stop_config = make_stop_config(max_num_outputs as u32, &eos_tokens1);
        let future = ctx1.generate_async(SamplerConfig::Greedy, &stop_config);
        let output = poll_generate(&future).await;

        println!("Output 1: {:?} (elapsed: {:?})", output, start.elapsed());
    };

    let ctx2 = common.fork();
    let eos_tokens2 = eos_tokens.clone();
    let handle2 = async {
        ctx2.fill_user("Explain the Espresso making process ELI5.");

        let stop_config = make_stop_config(max_num_outputs as u32, &eos_tokens2);
        let future = ctx2.generate_async(SamplerConfig::Greedy, &stop_config);
        let output = poll_generate(&future).await;

        println!("Output 2: {:?} (elapsed: {:?})", output, start.elapsed());
    };

    future::join(handle1, handle2).await;

    Ok(())
}
