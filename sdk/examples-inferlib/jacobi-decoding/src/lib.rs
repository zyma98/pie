//! Demonstrates Parallel Jacobi Decoding (PJD) for speculative generation.
//!
//! This example implements the PJD algorithm for speculative decoding. It
//! speculates multiple tokens in parallel, verifies them against the model's
//! actual predictions, accepts correct speculations, and refines incorrect
//! ones in subsequent iterations.
//!
//! This port uses the raw `Queue` and `ForwardPass` APIs to implement the
//! Jacobi decoding loop, bypassing `Context` for generation. KV cache pages
//! and position tracking are managed manually.

use inferlib_inference_bindings::{ChatFormatter, Model, Queue, Tokenizer};
use inferlib_run_bindings::{Args, Result};
use std::time::Instant;

const HELP: &str = "\
Usage: jacobi-decoding [OPTIONS]

Demonstrates Parallel Jacobi Decoding for speculative generation.

Options:
  -m, --max-tokens <N>          Maximum number of tokens to generate [default: 512]
  -g, --speculation-length <N>  Number of tokens to speculate in parallel [default: 5]
  -p, --prompt <PROMPT>         Prompt to generate
                                [default: Explain the LLM decoding process ELI5.]
  -h, --help                    Prints help information";

/// Grow the KV cache to accommodate `num_tokens` additional tokens.
fn grow_kv_pages(
    queue: &Queue,
    kv_page_ptrs: &mut Vec<u32>,
    kv_page_last_len: &mut u32,
    kv_page_size: u32,
    num_tokens: usize,
) {
    if num_tokens == 0 {
        return;
    }
    let current_total = if kv_page_ptrs.is_empty() {
        0u32
    } else {
        (kv_page_ptrs.len() as u32 - 1) * kv_page_size + *kv_page_last_len
    };
    let new_total = current_total + num_tokens as u32;
    let new_num_pages = (new_total + kv_page_size - 1) / kv_page_size;
    let pages_to_add = new_num_pages as usize - kv_page_ptrs.len();
    if pages_to_add > 0 {
        let new_pages = queue.allocate_kv_pages(pages_to_add as u32);
        kv_page_ptrs.extend(new_pages);
    }
    *kv_page_last_len = new_total % kv_page_size;
    if *kv_page_last_len == 0 {
        *kv_page_last_len = kv_page_size;
    }
}

/// Shrink the KV cache by removing the last `num_tokens` tokens worth of pages.
fn shrink_kv_pages(
    queue: &Queue,
    kv_page_ptrs: &mut Vec<u32>,
    kv_page_last_len: &mut u32,
    kv_page_size: u32,
    num_tokens: usize,
) {
    if num_tokens == 0 {
        return;
    }
    let current_total = if kv_page_ptrs.is_empty() {
        0u32
    } else {
        (kv_page_ptrs.len() as u32 - 1) * kv_page_size + *kv_page_last_len
    };
    let new_total = current_total.saturating_sub(num_tokens as u32);
    if new_total == 0 {
        let ptrs: Vec<u32> = kv_page_ptrs.drain(..).collect();
        queue.deallocate_kv_pages(&ptrs);
        *kv_page_last_len = 0;
        return;
    }
    let new_num_pages = (new_total + kv_page_size - 1) / kv_page_size;
    let pages_to_remove = kv_page_ptrs.len() as u32 - new_num_pages;
    if pages_to_remove > 0 {
        let removed: Vec<u32> = kv_page_ptrs.drain(new_num_pages as usize..).collect();
        queue.deallocate_kv_pages(&removed);
    }
    *kv_page_last_len = new_total % kv_page_size;
    if *kv_page_last_len == 0 {
        *kv_page_last_len = kv_page_size;
    }
}

/// Creates a causal attention mask in BRLE format.
///
/// Each input token at index `i` can attend to all tokens before it in the
/// context. The BRLE encoding `[n]` represents `n` unmasked (attendable)
/// positions.
fn causal_mask(num_total_tokens: u32, num_input_tokens: u32) -> Vec<Vec<u32>> {
    let offset = num_total_tokens - num_input_tokens;
    (0..num_input_tokens)
        .map(|i| vec![offset + i + 1])
        .collect()
}

/// Checks whether generation should stop based on token count or EOS sequences.
fn check_stop(tokens: &[u32], max_len: usize, eos_sequences: &[Vec<u32>]) -> bool {
    if tokens.len() >= max_len {
        return true;
    }
    for eos in eos_sequences {
        if tokens.len() >= eos.len() && tokens[tokens.len() - eos.len()..] == eos[..] {
            return true;
        }
    }
    false
}

/// Generates text using Parallel Jacobi Decoding.
///
/// Operates directly on the `Queue` and `ForwardPass` APIs, managing KV cache
/// pages and position IDs manually.
fn generate_with_pjd(
    queue: &Queue,
    tokenizer: &Tokenizer,
    prompt_tokens: Vec<u32>,
    gamma: usize,
    unk_token_id: u32,
    max_tokens: usize,
    eos_sequences: &[Vec<u32>],
    kv_page_size: u32,
) -> (String, usize) {
    let mut all_generated_tokens = Vec::new();
    let mut num_steps = 0;

    // KV cache state, managed manually.
    let mut kv_page_ptrs: Vec<u32> = Vec::new();
    let mut kv_page_last_len: u32 = 0;

    // Tracks which tokens and positions have been committed to the KV cache.
    let mut token_ids: Vec<u32> = Vec::new();
    let mut position_ids: Vec<u32> = Vec::new();

    // The initial batch tokens are the prompt followed by `gamma` unknown tokens.
    let mut batch_tokens = prompt_tokens;
    batch_tokens.extend(std::iter::repeat(unk_token_id).take(gamma));

    loop {
        // Check stop condition before generating more tokens.
        if check_stop(&all_generated_tokens, max_tokens, eos_sequences) {
            break;
        }

        // Example shape of the batch tokens when gamma = 3:
        // [truth_0, truth_1, spec_0, spec_1, spec_2]
        //
        // The truth tokens are the tokens that are input by the user at the beginning of
        // the generation or are the accepted tokens from the previous loop iteration.
        // The number of truth tokens is variable in each loop iteration.
        //
        // The speculative tokens are the ones that will be validated in the upcoming
        // forward pass. There will always be `gamma` speculative tokens.

        let batch_len = batch_tokens.len();

        // Calculate the positions for the tokens in the batch.
        let batch_positions = {
            let pos_offset = position_ids.last().map(|&p| p + 1).unwrap_or(0);
            (pos_offset..pos_offset + batch_len as u32).collect::<Vec<u32>>()
        };

        // Adjust the KV cache pages to accommodate the tokens in the batch.
        grow_kv_pages(
            queue,
            &mut kv_page_ptrs,
            &mut kv_page_last_len,
            kv_page_size,
            batch_len,
        );

        // The total length of the context for the next forward pass is the sum of the
        // number of tokens already in the KV cache and the number of tokens in the batch
        // waiting to be filled into the KV cache.
        let total_ctx_len = token_ids.len() + batch_len;

        // Each of the tokens in the batch will attend to the tokens before it in the context.
        let masks_for_batch = causal_mask(total_ctx_len as u32, batch_len as u32);

        // We sample at the position of the last truth token and the `gamma` speculative tokens.
        let sample_indices: Vec<u32> =
            ((batch_tokens.len() - gamma - 1) as u32..batch_tokens.len() as u32).collect();

        // Run a single forward pass to get sampled tokens for all `gamma + 1` positions.
        // We use temperature=0 for greedy sampling.
        let p = queue.create_forward_pass();
        p.input_tokens(&batch_tokens, &batch_positions);
        p.kv_cache(&kv_page_ptrs, kv_page_last_len);
        p.attention_mask(&masks_for_batch);
        p.output_tokens(&sample_indices, 0.0);
        let pass_result = p.execute();
        let sampled_tokens = pass_result.tokens.unwrap_or_default();

        // Create a slice over the speculative tokens that were fed into the forward pass.
        let speculated_tokens = &batch_tokens[batch_tokens.len() - gamma..];

        // Input batch:    [truth_0, truth_1, spec_0, spec_1, spec_2]
        //                              ↓       ↓       ↓       ↓
        // Sampled tokens: [N/A,      samp_0, samp_1, samp_2, samp_3]
        //
        // samp_0 is the ground truth next token in the generation process.
        // If the speculated token spec_0 is the same as the sampled token samp_0,
        // our speculation is correct, and the samp_1 token will also be correct.
        // Iteratively, if spec_1 is the same as samp_1, then samp_2 will also be correct,
        // and so on.
        //
        // The correct tokens go into the accepted tokens vector. Upon detecting the first
        // mismatch, all following sampled tokens go into the rejected tokens vector.
        let mut accepted_tokens = vec![sampled_tokens[0]];
        let mut rejected_tokens = vec![];
        for i in 0..gamma {
            if sampled_tokens[i] == speculated_tokens[i] {
                accepted_tokens.push(sampled_tokens[i + 1]);
            } else {
                rejected_tokens.extend_from_slice(&sampled_tokens[i + 1..]);
                break;
            }
        }

        // During the forward pass, all tokens in the batch are stored in the KV cache.
        // We need to shrink the KV cache to remove the wrong speculative tokens. The
        // number of wrong speculative tokens is the length of the rejected tokens vector.
        shrink_kv_pages(
            queue,
            &mut kv_page_ptrs,
            &mut kv_page_last_len,
            kv_page_size,
            rejected_tokens.len(),
        );

        // Update our internal state. The token_ids and position_ids reflect the tokens
        // that have already been stored in the KV cache. We extend these with the truth
        // tokens and then the accepted tokens in this loop iteration.
        token_ids.extend_from_slice(&batch_tokens[..batch_len - gamma]);
        token_ids.extend_from_slice(&accepted_tokens[..accepted_tokens.len() - 1]);
        position_ids.extend_from_slice(&batch_positions[..batch_len - gamma]);
        position_ids.extend_from_slice(
            &batch_positions[batch_len - gamma..batch_len - rejected_tokens.len()],
        );

        // Add the accepted tokens to the generated tokens vector.
        all_generated_tokens.extend_from_slice(&accepted_tokens);

        // The next batch will include the accepted tokens at the beginning.
        batch_tokens = accepted_tokens;

        // The next batch will also contain the sampled tokens from the previously
        // incorrect speculation. They are being "refined" in each loop iteration,
        // and hopefully in the next loop iteration, they will be correct. We pad
        // additional unknown tokens to the end of the batch to keep the speculation
        // length `gamma`.
        let add_unk_token_num = gamma - rejected_tokens.len();
        batch_tokens.extend(rejected_tokens.into_iter());
        batch_tokens.extend(std::iter::repeat(unk_token_id).take(add_unk_token_num));

        num_steps += 1;
    }

    // Return the generated tokens and the number of steps taken.
    (tokenizer.detokenize(&all_generated_tokens), num_steps)
}

#[inferlib_macros::main]
async fn main(mut args: Args) -> Result<()> {
    if args.contains(["-h", "--help"]) {
        println!("{}", HELP);
        return Ok(());
    }

    let max_num_outputs: usize = args.value_from_str(["-m", "--max-tokens"]).unwrap_or(512);
    let speculation_length: usize = args
        .value_from_str(["-s", "--speculation-length"])
        .unwrap_or(5);
    let prompt: String = args
        .value_from_str(["-p", "--prompt"])
        .unwrap_or_else(|_| "Explain the LLM decoding process ELI5.".to_string());

    let start = Instant::now();

    let model = Model::get_auto();
    let eos_tokens = model.eos_tokens();
    let tokenizer = model.get_tokenizer();
    let kv_page_size = model.get_kv_page_size();

    // Use the first EOS token as the UNK token ID.
    let unk_token_id = eos_tokens[0][0];

    // Build the prompt using ChatFormatter for proper template handling.
    let template = model.get_prompt_template();
    let formatter = ChatFormatter::new(&template);
    formatter.add_system("You are a helpful, respectful and honest assistant.");
    formatter.add_user(&prompt);
    let formatted = formatter.render(true, true);
    let prompt_tokens = tokenizer.tokenize(&formatted);

    // Create a queue for manual forward passes, bypassing Context.
    let queue = Queue::from_model_name(&model.get_name());

    println!(
        "Starting generation with Parallel Jacobi Decoding (speculation length = {})...",
        speculation_length
    );

    let (output, num_steps) = generate_with_pjd(
        &queue,
        &tokenizer,
        prompt_tokens,
        speculation_length,
        unk_token_id,
        max_num_outputs,
        &eos_tokens,
        kv_page_size,
    );

    let elapsed = start.elapsed();
    let output_token_ids = tokenizer.tokenize(&output);

    println!("\n--- Output ---\n{}\n--------------", output);

    println!(
        "Total elapsed: {:?}, Tokens generated: {}, Mean accepted tokens per step: {:.4}",
        elapsed,
        output_token_ids.len(),
        output_token_ids.len() as f64 / num_steps as f64
    );

    // Compute per-token latency
    if !output_token_ids.is_empty() {
        println!(
            "Per-token latency: {:?}",
            elapsed / output_token_ids.len() as u32
        );
    }

    Ok(())
}
