//! A custom sampler that constrains token generation to match a Lark grammar.
//!
//! Uses `GrammarMatcher` and `TokenMask` from `inferlib-llguidance` (linked at
//! runtime as a Wasm component) to compute token masks based on grammar state,
//! ensuring outputs are always syntactically valid. Called from the manual decode
//! loop in `lib.rs` via `Context::decode_step_dist`.

use inferlib_llguidance_bindings::GrammarMatcher;
use std::cell::RefCell;

pub struct ConstrainedSampler {
    inner: RefCell<Inner>,
}

struct Inner {
    matcher: GrammarMatcher,
    eos_token_id: u32,
}

type Vocab = (Vec<u32>, Vec<Vec<u8>>);

impl ConstrainedSampler {
    pub fn new(
        vocab: Vocab,
        special_tokens: Vocab,
        split_regex: String,
        grammar: String,
        eos_token_id: u32,
        escape_non_printable: bool,
    ) -> Self {
        let (vocab_ids, vocab_bytes) = vocab;
        let (special_token_ids, special_token_bytes) = special_tokens;

        let matcher = GrammarMatcher::new(
            &vocab_ids,
            &vocab_bytes,
            &special_token_ids,
            &special_token_bytes,
            &split_regex,
            &grammar,
            eos_token_id,
            escape_non_printable,
        );

        ConstrainedSampler {
            inner: RefCell::new(Inner {
                matcher,
                eos_token_id,
            }),
        }
    }

    /// Sample a token from the given distribution, constrained by the grammar.
    pub fn sample(&self, token_ids: &[u32], probs: &[f32]) -> u32 {
        let inner = self.inner.borrow_mut();

        let mask = match inner.matcher.compute_mask() {
            Some(m) => m,
            None => return inner.eos_token_id,
        };

        if mask.is_empty() {
            return inner.eos_token_id;
        }

        let mut max_prob = f32::NEG_INFINITY;
        let mut best_token = None;

        // Find the highest-probability token allowed by the grammar mask
        for (i, &token_id) in token_ids.iter().enumerate() {
            if mask.is_allowed(token_id) && probs[i] > max_prob {
                max_prob = probs[i];
                best_token = Some(token_id);
            }
        }

        let sampled_token_id = match best_token {
            Some(token) => token,
            None => return mask.first_bit_set().unwrap_or(0),
        };

        // Commit the chosen token to advance the parser state
        inner.matcher.consume_token(sampled_token_id);

        sampled_token_id
    }
}
