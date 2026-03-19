//! Prompt Lookup Decoding (PLD) implementation for speculative decoding.
//!
//! This module implements the same `inferlib:context/inference` interface as the
//! standard `context` library, but uses Prompt Lookup Decoding for faster generation.
//!
//! PLD works by:
//! 1. Finding n-gram matches between recent tokens and earlier parts of the context
//! 2. Speculatively proposing tokens that followed those matches
//! 3. Verifying all speculated tokens in a single forward pass
//! 4. Accepting tokens up to the first mismatch

// Generate WIT bindings for exports
wit_bindgen::generate!({
    path: "wit",
    world: "context-provider",
    with: {
        "inferlib:model/models": inferlib_model_bindings::model::models,
        "inferlib:queue/queues": inferlib_queue_bindings::queue::queues,
        "inferlib:chat/formatter": inferlib_chat_bindings::chat::formatter,
        "inferlib:brle/encoding": inferlib_brle_bindings::encoding,
    },
});

use exports::inferlib::context::inference::{Guest, GuestContext, SamplerConfig, StopConfig};

use inferlib_brle_bindings::Brle;
use inferlib_chat_bindings::ChatFormatter;
use inferlib_model_bindings::{Model as WitModel, Tokenizer};
use inferlib_queue_bindings::Queue as WitQueue;

use std::cell::RefCell;
use std::cmp::Ordering;
use std::mem;
use std::rc::Rc;

/// PLD configuration parameters
const NGRAM_SIZE: usize = 2; // Size of n-gram to match
const MAX_SPECULATION_LENGTH: usize = 10; // Maximum tokens to speculate ahead

struct InferenceImpl;

impl Guest for InferenceImpl {
    type Context = ContextImpl;
}

/// Internal Context struct implementing Prompt Lookup Decoding
struct Context {
    queue: WitQueue,
    tokenizer: Tokenizer,
    formatter: ChatFormatter,

    token_ids: Vec<u32>,
    token_ids_pending: Vec<u32>,

    token_mask_pending: Vec<Brle>,
    token_mask_current: Brle,

    position_ids: Vec<u32>,

    kv_page_ptrs: Vec<u32>,
    kv_page_last_len: usize,
    kv_page_size: usize,

    begin_of_sequence: bool,
}

impl Context {
    pub fn new(wit_model: &WitModel) -> Self {
        let model_name = wit_model.get_name();
        let queue = WitQueue::from_model_name(&model_name);
        let kv_page_size = wit_model.get_kv_page_size() as usize;
        let prompt_template = wit_model.get_prompt_template();

        let tokenizer = wit_model.get_tokenizer();
        let formatter =
            ChatFormatter::new(&prompt_template).expect("Failed to create chat formatter");

        Context {
            queue,
            tokenizer,
            formatter,
            token_ids: Vec::new(),
            token_ids_pending: Vec::new(),
            token_mask_pending: Vec::new(),
            token_mask_current: Brle::new(0),
            position_ids: Vec::new(),
            kv_page_ptrs: Vec::new(),
            kv_page_last_len: 0,
            kv_page_size,
            begin_of_sequence: true,
        }
    }

    pub fn get_token_ids(&self) -> &[u32] {
        &self.token_ids
    }

    pub fn get_text(&self) -> String {
        self.tokenizer.detokenize(&self.token_ids)
    }

    pub fn fill(&mut self, text: &str) {
        let new_token_ids = self.tokenizer.tokenize(text);
        self.fill_tokens(new_token_ids);
    }

    pub fn fill_tokens(&mut self, new_token_ids: Vec<u32>) {
        let n = new_token_ids.len();
        self.token_ids_pending.extend(new_token_ids);

        for _ in 0..n {
            self.token_mask_current.append(false);
            self.token_mask_pending
                .push(self.token_mask_current.clone())
        }
        self.begin_of_sequence = false;
    }

    pub fn fill_system(&mut self, text: &str) {
        self.formatter.add_system(text);
        self.flush_chat_messages(false);
    }

    pub fn fill_user(&mut self, text: &str) {
        self.formatter.add_user(text);
        self.flush_chat_messages(true);
    }

    pub fn fill_assistant(&mut self, text: &str) {
        self.formatter.add_assistant(text);
        self.flush_chat_messages(false);
    }

    fn flush_chat_messages(&mut self, add_generation_prompt: bool) {
        if self.formatter.has_messages() {
            let p = self
                .formatter
                .render(add_generation_prompt, self.begin_of_sequence);
            self.begin_of_sequence = false;
            self.formatter.clear();
            self.fill(&p);
        }
    }

    fn adjust_kv_pages(&mut self, num_tokens: isize) {
        if num_tokens == 0 {
            return;
        }

        let current_tokens = if self.kv_page_ptrs.is_empty() {
            self.kv_page_last_len
        } else {
            (self.kv_page_ptrs.len() - 1) * self.kv_page_size + self.kv_page_last_len
        };

        let new_total_tokens = match current_tokens.checked_add_signed(num_tokens) {
            Some(n) => n,
            None => panic!("Token count adjustment resulted in underflow"),
        };

        let current_pages = self.kv_page_ptrs.len();
        let required_pages = new_total_tokens.div_ceil(self.kv_page_size);

        match required_pages.cmp(&current_pages) {
            Ordering::Greater => {
                let new_pages_needed = required_pages - current_pages;
                let new_kv_page_ptrs = self.queue.allocate_kv_pages(new_pages_needed as u32);
                self.kv_page_ptrs.extend(new_kv_page_ptrs);
            }
            Ordering::Less => {
                let pages_to_free = self.kv_page_ptrs.split_off(required_pages);
                if !pages_to_free.is_empty() {
                    self.queue.deallocate_kv_pages(&pages_to_free);
                }
            }
            Ordering::Equal => {}
        }

        let last_page_len = new_total_tokens % self.kv_page_size;
        self.kv_page_last_len = if last_page_len == 0 && new_total_tokens > 0 {
            self.kv_page_size
        } else {
            last_page_len
        };
    }

    pub fn grow_kv_pages(&mut self, num_tokens: usize) {
        self.adjust_kv_pages(num_tokens as isize);
    }

    pub fn shrink_kv_pages(&mut self, num_tokens: usize) {
        self.adjust_kv_pages(-(num_tokens as isize));
    }

    pub fn flush(&mut self) {
        if self.token_ids_pending.is_empty() {
            return;
        }
        let process_count = self.token_ids_pending.len();

        let pending_token_ids = self
            .token_ids_pending
            .drain(..process_count)
            .collect::<Vec<u32>>();

        let mask = self
            .token_mask_pending
            .drain(..process_count)
            .map(|b| b.get_buffer())
            .collect::<Vec<Vec<u32>>>();

        let last_pos = self.position_ids.last().map(|&p| p + 1).unwrap_or(0);
        let position_ids =
            (last_pos..(last_pos + pending_token_ids.len() as u32)).collect::<Vec<u32>>();

        self.grow_kv_pages(pending_token_ids.len());

        let p = self.queue.create_forward_pass();
        p.input_tokens(&pending_token_ids, &position_ids);
        p.kv_cache(&self.kv_page_ptrs, self.kv_page_last_len as u32);
        p.attention_mask(&mask);

        let _ = p.execute();

        self.token_ids.extend(pending_token_ids);
        self.position_ids.extend(&position_ids);
    }

    /// Find n-gram matches in the existing context and return candidate continuation tokens.
    /// Returns None if no match found or not enough context.
    fn find_ngram_match(&self, ngram_size: usize, max_length: usize) -> Option<Vec<u32>> {
        // Combine committed and pending tokens to form the full context
        let all_tokens: Vec<u32> = self
            .token_ids
            .iter()
            .chain(self.token_ids_pending.iter())
            .copied()
            .collect();

        // Need at least ngram_size tokens to form an n-gram
        if all_tokens.len() < ngram_size {
            return None;
        }

        // The n-gram we're looking for (last ngram_size tokens)
        let ngram = &all_tokens[all_tokens.len() - ngram_size..];

        // Search for this n-gram in earlier parts of the context
        // We search from the beginning up to (but not including) the current n-gram position
        let search_end = all_tokens.len() - ngram_size;

        for start_pos in 0..search_end {
            // Check if n-gram matches at this position
            if all_tokens[start_pos..start_pos + ngram_size] == *ngram {
                // Found a match! Extract the tokens that followed
                let continuation_start = start_pos + ngram_size;
                let continuation_end = (continuation_start + max_length).min(search_end);

                if continuation_start < continuation_end {
                    return Some(all_tokens[continuation_start..continuation_end].to_vec());
                }
            }
        }

        None
    }

    /// Performs a speculative decoding step using Prompt Lookup Decoding.
    /// Returns the number of tokens that were successfully generated.
    fn speculative_decode_step(&mut self, sampler: &SamplerConfig) -> Vec<u32> {
        assert!(
            !self.token_ids_pending.is_empty(),
            "Must have at least one seed token"
        );

        // Try to find speculative tokens via n-gram lookup
        let speculative_tokens = self.find_ngram_match(NGRAM_SIZE, MAX_SPECULATION_LENGTH);

        match speculative_tokens {
            Some(draft_tokens) if !draft_tokens.is_empty() => {
                // We have draft tokens to verify
                self.verify_speculative_tokens(sampler, &draft_tokens)
            }
            _ => {
                // No speculation possible, fall back to single-token generation
                let token = self.decode_step_single(sampler);
                vec![token]
            }
        }
    }

    /// Verify speculative tokens and return accepted tokens plus the first new token.
    fn verify_speculative_tokens(
        &mut self,
        sampler: &SamplerConfig,
        draft_tokens: &[u32],
    ) -> Vec<u32> {
        // Take pending tokens
        let mut all_input_tokens = mem::take(&mut self.token_ids_pending);
        let mut all_masks = mem::take(&mut self.token_mask_pending);

        let original_pending_len = all_input_tokens.len();

        // Save the current mask length so we can roll back if needed
        let mask_len_before_drafts = self.token_mask_current.len();

        // Add draft tokens to pending (they need masks too)
        for &token in draft_tokens {
            self.token_mask_current.append(false);
            all_masks.push(self.token_mask_current.clone());
            all_input_tokens.push(token);
        }

        let total_tokens = all_input_tokens.len();

        // Compute position IDs
        let last_pos_id = self.position_ids.last().map(|&p| p + 1).unwrap_or(0);
        let position_ids = (last_pos_id..(last_pos_id + total_tokens as u32)).collect::<Vec<u32>>();

        // Grow KV pages for all tokens
        self.grow_kv_pages(total_tokens);

        let mask = all_masks
            .into_iter()
            .map(|brie| brie.get_buffer())
            .collect::<Vec<Vec<u32>>>();

        // Create forward pass requesting output at all positions
        let p = self.queue.create_forward_pass();
        p.input_tokens(&all_input_tokens, &position_ids);
        p.kv_cache(&self.kv_page_ptrs, self.kv_page_last_len as u32);
        p.attention_mask(&mask);

        // Request tokens at all positions (to verify drafts and get next token)
        let output_indices: Vec<u32> = (0..total_tokens as u32).collect();
        match sampler {
            SamplerConfig::Greedy => {
                p.output_tokens(&output_indices, 0.0);
            }
            SamplerConfig::Multinomial(temperature) => {
                p.output_tokens(&output_indices, *temperature);
            }
            SamplerConfig::TopP((temperature, top_p)) => {
                p.output_tokens_top_p(&output_indices, *temperature, *top_p);
            }
            SamplerConfig::TopK((temperature, top_k)) => {
                p.output_tokens_top_k(&output_indices, *temperature, *top_k);
            }
            SamplerConfig::MinP((temperature, min_p)) => {
                p.output_tokens_min_p(&output_indices, *temperature, *min_p);
            }
            SamplerConfig::TopKTopP((temperature, top_k, top_p)) => {
                p.output_tokens_top_k_top_p(&output_indices, *temperature, *top_k, *top_p);
            }
        }

        let res = p.execute();
        let sampled_tokens = res.tokens.unwrap();

        // Verify draft tokens: for each draft position, check if model agrees
        // Position i's output predicts position i+1's token
        // So sampled_tokens[original_pending_len - 1] predicts the first draft token
        // sampled_tokens[original_pending_len] predicts the second draft token, etc.

        let mut accepted_count = 0;
        for (i, &draft_token) in draft_tokens.iter().enumerate() {
            let prediction_idx = original_pending_len - 1 + i;
            if prediction_idx < sampled_tokens.len()
                && sampled_tokens[prediction_idx] == draft_token
            {
                accepted_count += 1;
            } else {
                break; // First mismatch, stop accepting
            }
        }

        // Build result: original pending tokens + accepted draft tokens + one new token
        let mut result = Vec::new();

        // The tokens we're committing from the input
        let commit_count = original_pending_len + accepted_count;

        // Commit the tokens to our state
        self.token_ids.extend(&all_input_tokens[..commit_count]);
        self.position_ids.extend(&position_ids[..commit_count]);

        // The new generated tokens are the draft tokens that were accepted
        for &draft_token in &draft_tokens[..accepted_count] {
            result.push(draft_token);
        }

        // Roll back token_mask_current to only include accepted tokens
        // We added len(draft_tokens) bits, but only accepted_count were valid
        let rejected_count = draft_tokens.len() - accepted_count;
        if rejected_count > 0 {
            // Remove the rejected bits from the end of the mask
            let new_mask_len = mask_len_before_drafts + accepted_count as u32;
            self.token_mask_current
                .remove_range(new_mask_len, self.token_mask_current.len());
        }

        // Get the next token (the one predicted at the last accepted position)
        let next_token_idx = original_pending_len - 1 + accepted_count;
        if next_token_idx < sampled_tokens.len() {
            let next_token = sampled_tokens[next_token_idx];
            result.push(next_token);

            // Add this token to pending for the next iteration
            self.token_mask_current.append(false);
            self.token_mask_pending
                .push(self.token_mask_current.clone());
            self.token_ids_pending.push(next_token);
        }

        // Shrink KV cache if we didn't use all the speculated positions
        let unused_tokens = total_tokens - commit_count;
        if unused_tokens > 0 {
            self.shrink_kv_pages(unused_tokens);
        }

        result
    }

    /// Single-token decode step (fallback when no speculation is possible)
    fn decode_step_single(&mut self, sampler: &SamplerConfig) -> u32 {
        let pending_token_ids = mem::take(&mut self.token_ids_pending);
        let last_pos_id = self.position_ids.last().map(|&p| p + 1).unwrap_or(0);
        let position_ids =
            (last_pos_id..(last_pos_id + pending_token_ids.len() as u32)).collect::<Vec<u32>>();

        self.grow_kv_pages(pending_token_ids.len());

        let mask = mem::take(&mut self.token_mask_pending)
            .into_iter()
            .map(|brie| brie.get_buffer())
            .collect::<Vec<Vec<u32>>>();

        let p = self.queue.create_forward_pass();

        p.input_tokens(&pending_token_ids, &position_ids);
        p.kv_cache(&self.kv_page_ptrs, self.kv_page_last_len as u32);
        p.attention_mask(&mask);

        let output_idx = pending_token_ids.len() as u32 - 1;
        match sampler {
            SamplerConfig::Greedy => {
                p.output_tokens(&[output_idx], 0.0);
            }
            SamplerConfig::Multinomial(temperature) => {
                p.output_tokens(&[output_idx], *temperature);
            }
            SamplerConfig::TopP((temperature, top_p)) => {
                p.output_tokens_top_p(&[output_idx], *temperature, *top_p);
            }
            SamplerConfig::TopK((temperature, top_k)) => {
                p.output_tokens_top_k(&[output_idx], *temperature, *top_k);
            }
            SamplerConfig::MinP((temperature, min_p)) => {
                p.output_tokens_min_p(&[output_idx], *temperature, *min_p);
            }
            SamplerConfig::TopKTopP((temperature, top_k, top_p)) => {
                p.output_tokens_top_k_top_p(&[output_idx], *temperature, *top_k, *top_p);
            }
        }

        let res = p.execute();
        let sampled = res.tokens.unwrap().into_iter().next().unwrap();

        // Commit the pending tokens
        self.token_ids.extend(pending_token_ids);
        self.position_ids.extend(position_ids);

        // Add the generated token to pending for the next iteration
        self.token_mask_current.append(false);
        self.token_mask_pending
            .push(self.token_mask_current.clone());
        self.token_ids_pending.push(sampled);

        sampled
    }

    /// Generates text using Prompt Lookup Decoding.
    pub fn generate(&mut self, sampler: &SamplerConfig, stop_config: &StopConfig) -> String {
        let mut generated_token_ids = Vec::new();

        loop {
            // Use speculative decoding step
            let new_tokens = self.speculative_decode_step(sampler);

            for &token in &new_tokens {
                generated_token_ids.push(token);

                // Check stop conditions after each token
                let should_stop = generated_token_ids.len() >= stop_config.max_tokens as usize
                    || stop_config
                        .eos_sequences
                        .iter()
                        .any(|seq| generated_token_ids.ends_with(seq));

                if should_stop {
                    return self.tokenizer.detokenize(&generated_token_ids);
                }
            }

            // If speculative_decode_step returned empty (shouldn't happen), break
            if new_tokens.is_empty() {
                break;
            }
        }

        self.tokenizer.detokenize(&generated_token_ids)
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        if !self.kv_page_ptrs.is_empty() {
            self.queue.deallocate_kv_pages(&self.kv_page_ptrs);
        }
    }
}

// WIT interface wrapper
struct ContextImpl {
    inner: Rc<RefCell<Context>>,
}

impl GuestContext for ContextImpl {
    fn new(wit_model: &WitModel) -> Self {
        let inner = Context::new(wit_model);
        ContextImpl {
            inner: Rc::new(RefCell::new(inner)),
        }
    }

    fn fill(&self, text: String) {
        self.inner.borrow_mut().fill(&text);
    }

    fn fill_system(&self, text: String) {
        self.inner.borrow_mut().fill_system(&text);
    }

    fn fill_user(&self, text: String) {
        self.inner.borrow_mut().fill_user(&text);
    }

    fn fill_assistant(&self, text: String) {
        self.inner.borrow_mut().fill_assistant(&text);
    }

    fn generate(&self, sampler_config: SamplerConfig, stop_config: StopConfig) -> String {
        self.inner
            .borrow_mut()
            .generate(&sampler_config, &stop_config)
    }

    fn flush(&self) {
        self.inner.borrow_mut().flush();
    }

    fn get_text(&self) -> String {
        self.inner.borrow().get_text()
    }

    fn get_token_ids(&self) -> Vec<u32> {
        self.inner.borrow().get_token_ids().to_vec()
    }
}

export!(InferenceImpl);
