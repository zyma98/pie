//! Demonstrates text watermarking for text generation.
//!
//! Uses a green/red list approach where tokens are partitioned based on the
//! hash of the previous token, and green-listed tokens receive a probability
//! boost during sampling. Uses `Context::decode_step_dist` to obtain per-step
//! token distributions, then applies the watermark bias before sampling.

use inferlib_inference_bindings::{Context, Model};
use inferlib_run_bindings::{Args, Result, anyhow};
use rand::SeedableRng;
use rand::distr::Distribution;
use rand::distr::weighted::WeightedIndex;
use rand::rngs::ThreadRng;
use rand::seq::SliceRandom;
use std::cell::RefCell;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::time::Instant;

/// Injects a watermark by partitioning the vocabulary into a "green list" and a "red list"
/// based on the hash of the previous token. It then boosts the probabilities of tokens
/// in the green list.
struct WatermarkSampler {
    inner: RefCell<Inner>,
}

struct Inner {
    /// The proportion of the vocabulary to be included in the green list (e.g., 0.5 for 50%).
    gamma: f32,
    /// The bias added to the logits of the green-listed tokens to increase their probability.
    delta: f32,
    /// The previously generated token ID, used to seed the green/red list generation.
    /// `None` indicates the start of a sequence.
    previous_token: Option<u32>,
    /// Random number generator for sampling.
    rng: ThreadRng,
}

impl WatermarkSampler {
    /// Creates a new `WatermarkSampler`.
    ///
    /// # Arguments
    /// * `gamma` - The proportion of the vocabulary to be in the green list (0.0 to 1.0).
    /// * `delta` - The bias to add to the logits of green-listed tokens.
    pub fn new(gamma: f32, delta: f32) -> Self {
        assert!(
            (0.0..=1.0).contains(&gamma),
            "gamma must be between 0.0 and 1.0."
        );
        Self {
            inner: RefCell::new(Inner {
                gamma,
                delta,
                previous_token: None,
                rng: rand::rng(),
            }),
        }
    }

    /// Samples a token from the watermarked distribution.
    ///
    /// Called with the next-token distribution from `Context::decode_step_dist`.
    /// Applies the green/red list bias and samples from the modified distribution.
    pub fn sample(&self, ids: &[u32], probs: &[f32]) -> u32 {
        let mut inner = self.inner.borrow_mut();

        if ids.is_empty() {
            inner.previous_token = Some(0);
            return 0;
        }

        let seed = inner.get_seed();
        let mut seeded_rng = rand::rngs::StdRng::seed_from_u64(seed);

        let mut watermarked_probs = probs.to_vec();

        let green_list_size = (ids.len() as f32 * inner.gamma).round() as usize;

        let mut indices: Vec<usize> = (0..ids.len()).collect();
        indices.shuffle(&mut seeded_rng);

        let green_indices: std::collections::HashSet<usize> =
            indices.into_iter().take(green_list_size).collect();

        // Apply the bias `delta` to the log-probabilities of the green-listed tokens.
        // logit = log(p), new_logit = log(p) + delta, new_p = p * exp(delta)
        let exp_delta = inner.delta.exp();
        for i in 0..watermarked_probs.len() {
            if green_indices.contains(&i) {
                watermarked_probs[i] *= exp_delta;
            }
        }

        let prob_sum: f32 = watermarked_probs.iter().sum();
        if prob_sum > 0.0 {
            for p in &mut watermarked_probs {
                *p /= prob_sum;
            }
        }

        let dist = WeightedIndex::new(&watermarked_probs)
            .expect("Failed to create watermarked distribution.");
        let chosen_idx = dist.sample(&mut inner.rng);

        let chosen_id = ids[chosen_idx];

        inner.previous_token = Some(chosen_id);

        chosen_id
    }
}

impl Inner {
    /// Hashes the previous token to create a seed for the RNG.
    /// This ensures the green/red list is deterministic based on the context.
    fn get_seed(&self) -> u64 {
        match self.previous_token {
            Some(token) => {
                let mut hasher = DefaultHasher::new();
                token.hash(&mut hasher);
                hasher.finish()
            }
            // Use a default seed for the very first token in a sequence.
            None => 0,
        }
    }
}

#[inferlib_macros::main]
async fn main(mut args: Args) -> Result<()> {
    let prompt = args
        .value_from_str(["-p", "--prompt"])
        .unwrap_or_else(|_| "Explain the LLM decoding process ELI5.".to_string());

    let max_num_outputs: usize = args.value_from_str(["-n", "--max-tokens"]).unwrap_or(256);

    let remaining = args.finish();
    if !remaining.is_empty() {
        return Err(anyhow!(
            "Unknown arguments found: {:?}. Use --help for usage.",
            remaining
        ));
    }

    let start = Instant::now();

    let model = Model::get_auto();
    let tokenizer = model.get_tokenizer();

    let eos_sequences = model.eos_tokens();
    let ctx = Context::new(&model);
    let watermark_sampler = WatermarkSampler::new(0.5, 0.0);

    ctx.fill_system("You are a helpful, respectful and honest assistant.");
    ctx.fill_user(&prompt);

    // Manual decode loop using decode_step_dist for watermarked sampling.
    // Each step gets the full next-token distribution, applies the watermark
    // bias via WatermarkSampler, and feeds the chosen token back.
    let mut generated_token_ids = Vec::new();
    loop {
        let dist = ctx.decode_step_dist(1.0, None);
        let token = watermark_sampler.sample(&dist.ids, &dist.probs);
        ctx.fill_token(token);
        generated_token_ids.push(token);

        if generated_token_ids.len() >= max_num_outputs
            || eos_sequences
                .iter()
                .any(|seq| generated_token_ids.ends_with(seq))
        {
            break;
        }
    }
    let text = tokenizer.detokenize(&generated_token_ids);

    println!("Output: {:?} (total elapsed: {:?})", text, start.elapsed());

    // Compute per-token latency, avoiding division by zero.
    if !generated_token_ids.is_empty() {
        println!(
            "Per token latency: {:?}",
            start.elapsed() / (generated_token_ids.len() as u32)
        );
    }

    Ok(())
}
