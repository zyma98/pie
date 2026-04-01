//! Demonstrates speculative decoding with a cache-based drafter (Cacheback Decoding).
//!
//! This variant uses the `inferlib-cacheback-py` Python library component which
//! manages the two-level LRU cache table, Trie-based draft organization, and
//! sliding-window state internally.  The main model verifies speculated tokens
//! via `Context::verify_draft` in a single batched forward pass with tree
//! attention.

use inferlib_cacheback_py_bindings::CacheTable;
use inferlib_inference_bindings::{Context, Model};
use inferlib_run_bindings::{Args, Result};
use std::time::Instant;

const HELP: &str = "\
Usage: cacheback-py-decoding [OPTIONS]

Demonstrates cacheback decoding with a Python cache-table component.

Options:
  -p, --prompt <STRING>    The prompt to send to the model
                           [default: Keep printing 'hello, world!' 100 times.]
  -n, --max-tokens <INT>   The maximum number of new tokens to generate [default: 256]
  -h, --help               Print help information";

const LEADER_CAPACITY: u32 = 256;
const FOLLOWER_CAPACITY: u32 = 4;
const LEADER_LEN: u32 = 1;
const FOLLOWER_LEN: u32 = 2;

#[inferlib_macros::main]
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

    let model = Model::get_auto();
    let eos_tokens = model.eos_tokens();
    let tokenizer = model.get_tokenizer();

    let ctx = Context::new(&model);
    ctx.fill_system("You are a helpful, respectful and honest assistant.");
    ctx.fill_user(&prompt);

    let cache_table = CacheTable::new(LEADER_CAPACITY, FOLLOWER_CAPACITY, LEADER_LEN, FOLLOWER_LEN);

    let mut all_generated_tokens = Vec::new();
    let mut num_tokens_per_step = Vec::new();

    println!("Starting generation with speculative decoding...");

    // Seed the cache table with prompt n-grams.  We cannot call
    // `ctx.get_token_ids()` before the first `verify_draft` because
    // `fill_system`/`fill_user` place tokens in a pending buffer that is only
    // flushed to `token_ids` during `verify_draft`.  So we run one iteration
    // with an empty draft first, then seed the cache with the now-available
    // prompt token IDs.
    {
        let result = cache_table.draft();
        let accepted = ctx.verify_draft(&result.tokens, &result.positions);
        cache_table.update(&ctx.get_token_ids());
        cache_table.update(&accepted);
        num_tokens_per_step.push(accepted.len());
        all_generated_tokens.extend_from_slice(&accepted);
    }

    loop {
        if all_generated_tokens.len() >= max_num_outputs
            || eos_tokens
                .iter()
                .any(|seq| all_generated_tokens.ends_with(seq))
        {
            break;
        }

        let result = cache_table.draft();
        let accepted = ctx.verify_draft(&result.tokens, &result.positions);

        cache_table.update(&accepted);
        num_tokens_per_step.push(accepted.len());
        all_generated_tokens.extend_from_slice(&accepted);

        if all_generated_tokens.len() >= max_num_outputs
            || eos_tokens
                .iter()
                .any(|seq| all_generated_tokens.ends_with(seq))
        {
            break;
        }
    }

    let output = tokenizer.detokenize(&all_generated_tokens);

    println!("Generation completed.");

    println!(
        "Output: {:?} (total elapsed: {:?})",
        output,
        start.elapsed()
    );

    if !all_generated_tokens.is_empty() {
        println!(
            "Per token latency: {:?}, Mean accepted tokens per step: {:.4}",
            start.elapsed() / (all_generated_tokens.len() as u32),
            num_tokens_per_step.iter().sum::<usize>() as f64 / num_tokens_per_step.len() as f64
        );
    }

    Ok(())
}
