//! Demonstrates speculative decoding with a cache-based drafter (Cacheback Decoding).
//!
//! This example uses a `CacheDrafter` that records token patterns from the
//! previous context and speculates future tokens based on n-gram matching. The main
//! model then verifies the speculated tokens, accepting matches and rejecting
//! mismatches.

mod cache_drafter;

use inferlet::sampler::Sample;
use inferlet::stop_condition::{self, StopCondition};
use inferlet::{Args, Result};
use std::cmp::Ordering;
use std::time::Instant;

const HELP: &str = "\
Usage: cacheback-decoding [OPTIONS]

Demonstrates cacheback decoding with a cache-based drafter.

Options:
  -p, --prompt <STRING>    The prompt to send to the model
                           [default: Keep printing 'hello, world!' 100 times.]
  -n, --max-tokens <INT>   The maximum number of new tokens to generate [default: 256]
  -h, --help               Print help information";

/// A simple greedy sampler that always picks the token with highest probability.
struct GreedySampler;

impl Sample for GreedySampler {
    fn sample(&self, ids: &[u32], probs: &[f32]) -> u32 {
        if ids.is_empty() {
            return 0;
        }

        let max_idx = probs
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Ordering::Equal))
            .map(|(idx, _)| idx)
            .unwrap_or(0);
        ids[max_idx]
    }
}

#[inferlet::main]
async fn main(mut args: Args) -> Result<()> {
    if args.contains(["-h", "--help"]) {
        println!("{}", HELP);
        return Ok(());
    }

    let prompt: String = args
        .value_from_str(["-p", "--prompt"])
        .unwrap_or_else(|_| "Keep printing 'hello, world!' 100 times.".to_string());
    let max_num_outputs: usize = args.value_from_str(["-n", "--max-tokens"]).unwrap_or(256);

    let start = Instant::now();

    let model = inferlet::get_auto_model();
    let eos_tokens = model.eos_tokens();

    let mut ctx = model.create_context();
    ctx.fill_system("You are a helpful, respectful and honest assistant.");
    ctx.fill_user(&prompt);

    let mut sampler = GreedySampler;
    let mut stop_condition =
        stop_condition::max_len(max_num_outputs).or(stop_condition::ends_with_any(eos_tokens));

    let mut drafter = cache_drafter::CacheDrafter::new(256, 4, 1, 2);

    let mut num_token_generated_per_step = Vec::new();

    println!("Starting generation with speculative decoding...");

    let output = ctx
        .generate_with_drafter(
            &mut drafter,
            &mut sampler,
            &mut stop_condition,
            Some(&mut num_token_generated_per_step),
        )
        .await;

    println!("Generation completed.");

    let output_token_ids = ctx.tokenizer.tokenize(&output);

    println!(
        "Output: {:?} (total elapsed: {:?})",
        output,
        start.elapsed()
    );

    if !output_token_ids.is_empty() {
        println!(
            "Per token latency: {:?}, Mean accepted tokens per step: {:.4}",
            start.elapsed() / (output_token_ids.len() as u32),
            num_token_generated_per_step.iter().sum::<usize>() as f64
                / num_token_generated_per_step.len() as f64
        );
    }

    Ok(())
}
