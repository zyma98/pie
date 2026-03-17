//! Demonstrates output validation by computing normalized probabilities over candidate strings.
//!
//! This example shows how to evaluate the likelihood of different candidate outputs
//! given a context. It uses the low-level ForwardPass API to obtain next-token
//! distributions, since `decode_step` only returns a sampled token ID.

use inferlib_inference_bindings::{Context, Model, Queue};
use inferlib_run_bindings::{Args, Result, anyhow};
use std::time::Instant;

const HELP: &str = "\
Usage: output-validation [OPTIONS]

A program to validate and rank candidate outputs based on their generation probability.

Options:
  -h, --help  Prints this help message";

/// Calculates the normalized probability of a list of candidate strings being generated from a given context.
///
/// This function evaluates each candidate string by calculating its conditional probability
/// given the initial state of the context. It then normalizes these probabilities so that they
/// sum to 1, providing a clear distribution of likelihood over the candidates.
///
/// Uses the ForwardPass API to obtain full next-token distributions, since the inferlib
/// Context only exposes `decode_step` (which returns a single sampled token, not a distribution).
///
/// # Arguments
/// * `model`: The model, used to create a queue and tokenizer.
/// * `ctx`: The initial context from which generation probabilities are calculated.
/// * `candidates`: A slice of strings representing the possible outputs to validate.
///
/// # Returns
/// A `Vec<(String, f32)>` where each tuple contains a candidate string and its normalized probability.
pub fn validate_outputs(model: &Model, ctx: &Context, candidates: &[String]) -> Vec<(String, f32)> {
    let tokenizer = model.get_tokenizer();
    let queue = Queue::from_model_name(&model.get_name());
    let mut log_probs = Vec::new();

    for candidate in candidates.iter() {
        let candidate_ctx = ctx.fork();
        let candidate_tokens = tokenizer.tokenize(candidate);
        let mut current_log_prob = 0.0f32;

        // Calculate the cumulative log probability for the candidate token sequence
        for &token_id in &candidate_tokens {
            candidate_ctx.flush();

            // Re-run the last committed token through a forward pass to obtain
            // the next-token distribution (the logits from flush are not stored).
            let fp = queue.create_forward_pass();
            let committed = candidate_ctx.get_token_ids();
            let last_token = *committed.last().unwrap();
            let position = (committed.len() - 1) as u32;

            fp.input_tokens(&[last_token], &[position]);
            fp.kv_cache(
                &candidate_ctx.get_kv_page_ptrs(),
                candidate_ctx.get_kv_page_last_len(),
            );
            fp.output_distributions(&[0], 1.0, None);
            let result = fp.execute();

            // Find the probability of the actual next token in our candidate
            if let Some(ref dists) = result.distributions {
                if let Some(dist) = dists.first() {
                    if let Some(index) = dist.ids.iter().position(|&id| id == token_id) {
                        let prob = dist.probs[index];
                        if prob > 0.0 {
                            current_log_prob += prob.ln();
                        } else {
                            current_log_prob = -1000.0;
                            break;
                        }
                    } else {
                        current_log_prob = -1000.0;
                        break;
                    }
                }
            }

            // Fill the context with the current token to prepare for the next step
            candidate_ctx.fill_token(token_id);
        }
        log_probs.push(current_log_prob);
    }

    // Normalize the probabilities
    // Find the maximum log probability for numerical stability (softmax trick)
    let max_log_prob = log_probs.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    if max_log_prob.is_infinite() {
        // If all probabilities are zero, return a uniform distribution.
        let uniform_prob = 1.0 / candidates.len() as f32;
        return candidates
            .iter()
            .map(|c| (c.clone(), uniform_prob))
            .collect();
    }

    // Convert log probabilities to standard probabilities and sum them up
    let mut total_prob = 0.0;
    let probs: Vec<f32> = log_probs
        .iter()
        .map(|&log_p| {
            let p = (log_p - max_log_prob).exp();
            total_prob += p;
            p
        })
        .collect();

    // Normalize to get the final distribution
    candidates
        .iter()
        .zip(probs.iter())
        .map(|(candidate, &p)| (candidate.clone(), p / total_prob))
        .collect()
}

#[inferlib_macros::main]
async fn main(mut args: Args) -> Result<()> {
    if args.contains(["-h", "--help"]) {
        println!("{}", HELP);
        return Ok(());
    }

    let start = Instant::now();
    let model = Model::get_auto();
    let ctx = Context::new(&model);

    if !model.get_name().starts_with("llama-3") {
        return Err(anyhow!(
            "Output validation example is only implemented for Llama 3 models. Got: {}",
            model.get_name()
        ));
    }

    // 1. Set up the initial context (the "prompt")
    let prompt = "The name of the person in the report is ";
    ctx.fill("<|begin_of_text|>");
    ctx.fill(
        "<|start_header_id|>system<|end_header_id|>\n\n\
        You are an expert at information extraction.<|eot_id|>",
    );
    ctx.fill(&format!(
        "<|start_header_id|>user<|end_header_id|>\n\n\
        From the sentence \"The financial report was prepared by David Chen.\", \
        extract the person's name.<|eot_id|>"
    ));
    ctx.fill("<|start_header_id|>assistant<|end_header_id|>\n\n");
    ctx.fill(prompt);
    ctx.flush();

    // 2. Define the list of candidate outputs to validate
    let candidates = vec![
        "John Smith".to_string(),
        "Mary Anne".to_string(),
        "David Chen".to_string(),
        "Chen David".to_string(),
    ];

    println!("--- Context ---\n'{}'\n\n--- Candidates ---", prompt);
    for c in &candidates {
        println!("- {}", c);
    }

    // 3. Call the validation function
    let results = validate_outputs(&model, &ctx, &candidates);

    println!("\n--- Validation Results ---");
    for (candidate, probability) in results {
        println!(
            "- Candidate: {:<12} | Probability: {:.4}%",
            candidate,
            probability * 100.0
        );
    }

    println!("\nTotal elapsed: {:?}", start.elapsed());

    Ok(())
}
