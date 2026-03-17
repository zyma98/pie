//! Generates text using a simple sliding window for KV cache management.
//!
//! This method keeps only the most recent `window_size` tokens in the KV cache.
//! As new tokens are generated, the oldest tokens beyond the window size are
//! masked and eventually evicted from the cache. This is the simplest form of
//! windowed attention, suitable for tasks where only recent context is relevant.

use inferlib_inference_bindings::{Context, Model, SamplerConfig, StopConfig, Tokenizer};
use inferlib_run_bindings::{Args, Result};
use std::time::Instant;

fn copy_sampler_config(s: &SamplerConfig) -> SamplerConfig {
    match s {
        SamplerConfig::Greedy => SamplerConfig::Greedy,
        SamplerConfig::Multinomial(t) => SamplerConfig::Multinomial(*t),
        SamplerConfig::TopP(v) => SamplerConfig::TopP(*v),
        SamplerConfig::TopK(v) => SamplerConfig::TopK(*v),
        SamplerConfig::MinP(v) => SamplerConfig::MinP(*v),
        SamplerConfig::TopKTopP(v) => SamplerConfig::TopKTopP(*v),
    }
}

fn check_stop(generated: &[u32], stop_config: &StopConfig) -> bool {
    if generated.len() >= stop_config.max_tokens as usize {
        return true;
    }
    for eos in &stop_config.eos_sequences {
        if generated.ends_with(eos) {
            return true;
        }
    }
    false
}

pub fn generate_with_sliding_window(
    ctx: &mut Context,
    sampler: &SamplerConfig,
    stop_config: &StopConfig,
    tokenizer: &Tokenizer,
    window_size: usize,
) -> String {
    let mut generated_token_ids = Vec::new();

    // The autoregressive generation loop
    loop {
        // 1. Decode the next token, sample, and add it to the pending buffer.
        let next_token_id = ctx.decode_step(copy_sampler_config(sampler));
        ctx.fill_token(next_token_id);
        generated_token_ids.push(next_token_id);

        // 2. Check for the stop condition.
        if check_stop(&generated_token_ids, stop_config) {
            break;
        }

        // 3. Apply sliding window logic.
        let committed_len = ctx.get_token_ids().len();
        if committed_len > window_size {
            // Mask all tokens from the beginning that are now outside the window.
            let evict_end = committed_len - window_size;
            ctx.mask_token_range(1, evict_end as u32, true);
            ctx.drop_masked_kv_pages();
        }
    }

    tokenizer.detokenize(&generated_token_ids)
}

#[inferlib_macros::main]
async fn main(mut args: Args) -> Result<()> {
    let prompt: String = args
        .value_from_str(["-p", "--prompt"])
        .unwrap_or("Explain LLM decoding process in ELI5.".to_string());
    let max_num_outputs: usize = args.value_from_str(["-n", "--max-tokens"]).unwrap_or(512);
    let window_size: usize = args.value_from_str(["-w", "--window-size"]).unwrap_or(32);

    let start = Instant::now();

    let model = Model::get_auto();
    let tokenizer = model.get_tokenizer();

    let mut ctx = Context::new(&model);
    ctx.fill_system("You are a helpful, respectful and honest assistant.");
    ctx.fill_user(&prompt);

    let sampler = SamplerConfig::Greedy;

    let stop_config = StopConfig {
        max_tokens: max_num_outputs as u32,
        eos_sequences: model.eos_tokens(),
    };

    println!("Starting generation with Windowed Attention (window_size={window_size})");

    let output =
        generate_with_sliding_window(&mut ctx, &sampler, &stop_config, &tokenizer, window_size);

    let elapsed = start.elapsed();
    let output_token_ids = tokenizer.tokenize(&output);

    println!("\n--- Output ---\n{}\n--------------", output);

    println!(
        "Total elapsed: {:?}, Tokens generated: {}",
        elapsed,
        output_token_ids.len()
    );

    // compute per token latency
    if !output_token_ids.is_empty() {
        println!(
            "Per-token latency: {:?}",
            elapsed / output_token_ids.len() as u32
        );
    }

    Ok(())
}
