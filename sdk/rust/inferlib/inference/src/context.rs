use crate::brle::Brle;
use crate::chat::ChatFormatter;
use crate::exports::inferlib::inference::inference::{
    GuestContext, GuestDecodeStepFuture, GuestFlushFuture, GuestGenerateFuture, SamplerConfig,
    StopConfig,
};
use crate::forward::KvPage;
use crate::models::{Model, ModelImpl, Tokenizer};
use crate::queues::Queue;

use inferlib_engine_bindings::inferlet::core::forward::ForwardPassResult as HostForwardPassResult;
use inferlib_engine_bindings::inferlet::core::runtime::get_model;

use std::cell::RefCell;
use std::cmp::Ordering;
use std::mem;
use std::rc::Rc;
use wstd::runtime::block_on;

fn greedy_argmax(ids: &[u32], probs: &[f32]) -> u32 {
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

pub(crate) struct Context {
    model: Model,
    queue: Queue,
    tokenizer: Tokenizer,
    formatter: ChatFormatter,

    token_ids: Vec<u32>,
    token_ids_pending: Vec<u32>,

    token_mask_pending: Vec<Brle>,
    token_mask_current: Brle,

    position_ids: Vec<u32>,

    kv_pages: Vec<KvPage>,
    kv_page_last_len: usize,
    kv_page_size: usize,

    adapter_ptr: Option<u32>,
    adapter_random_seed: Option<i64>,

    begin_of_sequence: bool,
}

impl Context {
    fn new(model: &Model) -> Self {
        let model_name = model.get_name();
        let host_model = get_model(&model_name).expect("Failed to get model");
        let queue = Queue::from_host_model(&host_model);
        let kv_page_size = model.get_kv_page_size() as usize;
        let prompt_template = model.get_prompt_template();
        let tokenizer = model.get_tokenizer();
        let formatter =
            ChatFormatter::new(prompt_template).expect("Failed to create chat formatter");

        Context {
            model: model.clone(),
            queue,
            tokenizer,
            formatter,
            token_ids: Vec::new(),
            token_ids_pending: Vec::new(),
            token_mask_pending: Vec::new(),
            token_mask_current: Brle::new(0),
            position_ids: Vec::new(),
            kv_pages: Vec::new(),
            kv_page_last_len: 0,
            kv_page_size,
            adapter_ptr: None,
            adapter_random_seed: None,
            begin_of_sequence: true,
        }
    }

    /// Creates a new Context from previously exported and now imported KV pages.
    /// This is used to restore a context's state from a cache.
    fn from_imported_state(
        model: &Model,
        kv_page_ptrs: Vec<u32>,
        prefix_tokens: Vec<u32>,
        kv_page_last_len: usize,
    ) -> Self {
        let model_name = model.get_name();
        let host_model = get_model(&model_name).expect("Failed to get model");
        let queue = Queue::from_host_model(&host_model);
        let kv_page_size = model.get_kv_page_size() as usize;
        let prompt_template = model.get_prompt_template();
        let tokenizer = model.get_tokenizer();
        let formatter =
            ChatFormatter::new(prompt_template).expect("Failed to create chat formatter");

        assert_eq!(
            prefix_tokens.len(),
            (kv_page_ptrs.len() - 1) * kv_page_size + kv_page_last_len,
        );

        let num_tokens = prefix_tokens.len();
        let kv_pages = kv_page_ptrs
            .into_iter()
            .map(|ptr| KvPage::new(&queue, ptr))
            .collect();

        Context {
            model: model.clone(),
            queue,
            tokenizer,
            formatter,
            token_ids: prefix_tokens,
            token_ids_pending: Vec::new(),
            token_mask_pending: Vec::new(),
            token_mask_current: Brle::new(num_tokens),
            position_ids: (0..num_tokens as u32).collect(),
            kv_pages,
            kv_page_last_len,
            kv_page_size,
            adapter_ptr: None,
            adapter_random_seed: None,
            begin_of_sequence: false,
        }
    }

    fn get_token_ids(&self) -> &[u32] {
        &self.token_ids
    }

    fn get_text(&self) -> String {
        self.tokenizer.detokenize(&self.token_ids)
    }

    /// Returns the unique IDs of the KV cache pages currently in use.
    fn get_kv_page_ptrs(&self) -> Vec<u32> {
        self.kv_pages.iter().map(|p| p.ptr()).collect()
    }

    /// Returns the number of tokens stored in the last KV cache page.
    fn get_kv_page_last_len(&self) -> usize {
        self.kv_page_last_len
    }

    fn fill(&mut self, text: &str) {
        let new_token_ids = self.tokenizer.tokenize(text);
        self.fill_tokens(new_token_ids);
    }

    fn fill_tokens(&mut self, new_token_ids: Vec<u32>) {
        let n = new_token_ids.len();
        self.token_ids_pending.extend(new_token_ids);

        for _ in 0..n {
            self.token_mask_current.append(false);
            self.token_mask_pending
                .push(self.token_mask_current.clone())
        }
        self.begin_of_sequence = false;
    }

    fn fill_token(&mut self, new_token_id: u32) {
        self.token_ids_pending.push(new_token_id);
        self.token_mask_current.append(false);
        self.token_mask_pending
            .push(self.token_mask_current.clone());
        self.begin_of_sequence = false;
    }

    fn fill_system(&mut self, text: &str) {
        self.formatter.add_system(text);
        self.flush_chat_messages(false);
    }

    fn fill_user(&mut self, text: &str) {
        self.formatter.add_user(text);
        self.flush_chat_messages(true);
    }

    fn fill_user_only(&mut self, text: &str) {
        self.formatter.add_user(text);
        self.flush_chat_messages(false);
    }

    fn fill_assistant(&mut self, text: &str) {
        self.formatter.add_assistant(text);
        self.flush_chat_messages(false);
    }

    fn mask_tokens(&mut self, indices: &[usize], mask: bool) {
        self.token_mask_current.mask(indices, mask)
    }

    fn mask_token_range(&mut self, start: usize, end: usize, mask: bool) {
        self.token_mask_current.mask_range(start, end, mask)
    }

    fn mask_token(&mut self, index: usize, mask: bool) {
        self.token_mask_current.mask(&[index], mask)
    }

    /// Drops fully masked KV pages to save memory, supporting non-contiguous
    /// dropping for optimizations like attention sink.
    ///
    /// Iterates through all committed pages and checks if the tokens corresponding
    /// to a page are all masked as `true`. If so, it removes the page (triggering
    /// deallocation via reference counting when the last reference is dropped) and
    /// removes the corresponding token ranges from the context's state.
    fn drop_masked_kv_pages(&mut self) {
        let num_committed_pages = self.token_ids.len() / self.kv_page_size;

        for i in (0..num_committed_pages).rev() {
            let page_start_token_idx = i * self.kv_page_size;
            let page_end_token_idx = (i + 1) * self.kv_page_size;

            if self.token_mask_current.is_range_all_value(
                page_start_token_idx,
                page_end_token_idx,
                true,
            ) {
                self.kv_pages.remove(i);

                self.token_ids
                    .drain(page_start_token_idx..page_end_token_idx);

                self.position_ids
                    .drain(page_start_token_idx..page_end_token_idx);

                self.token_mask_current
                    .remove_range(page_start_token_idx, page_end_token_idx);

                for mask in &mut self.token_mask_pending {
                    mask.remove_range(page_start_token_idx, page_end_token_idx);
                }
            }
        }

        let new_total_tokens = self.token_ids.len();
        let last_page_len = new_total_tokens % self.kv_page_size;

        self.kv_page_last_len = if last_page_len == 0 && new_total_tokens > 0 {
            self.kv_page_size
        } else {
            last_page_len
        };
    }

    fn set_adapter(&mut self, adapter_ptr: u32) {
        self.adapter_ptr = Some(adapter_ptr);
    }

    fn remove_adapter(&mut self) {
        self.adapter_ptr = None;
    }

    fn set_adapter_random_seed(&mut self, seed: i64) {
        self.adapter_random_seed = Some(seed);
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

    /// Adjusts the number of KV pages to match the required number of tokens.
    ///
    /// Handles both allocating new pages (growing) and deallocating unused pages
    /// (shrinking). A positive `num_tokens` grows the KV cache, while a negative
    /// value shrinks it. Deallocation of the underlying page happens via reference
    /// counting when the last `KvPage` reference is dropped.
    fn adjust_kv_pages(&mut self, num_tokens: isize) {
        if num_tokens == 0 {
            return;
        }

        let current_tokens = if self.kv_pages.is_empty() {
            self.kv_page_last_len
        } else {
            (self.kv_pages.len() - 1) * self.kv_page_size + self.kv_page_last_len
        };

        let new_total_tokens = match current_tokens.checked_add_signed(num_tokens) {
            Some(n) => n,
            None => panic!("Token count adjustment resulted in underflow"),
        };

        let current_pages = self.kv_pages.len();
        let required_pages = new_total_tokens.div_ceil(self.kv_page_size);

        match required_pages.cmp(&current_pages) {
            Ordering::Greater => {
                let new_pages_needed = required_pages - current_pages;
                let new_ptrs = self.queue.allocate_kv_pages(new_pages_needed as u32);
                let new_kv_pages = new_ptrs
                    .into_iter()
                    .map(|ptr| KvPage::new(&self.queue, ptr))
                    .collect::<Vec<_>>();
                self.kv_pages.extend(new_kv_pages);
            }
            Ordering::Less => {
                // Truncate drops the removed KvPage objects, which will deallocate
                // the underlying pages only if no other context holds a reference.
                self.kv_pages.truncate(required_pages);
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

    fn grow_kv_pages(&mut self, num_tokens: usize) {
        self.adjust_kv_pages(num_tokens as isize);
    }

    #[allow(dead_code)]
    fn shrink_kv_pages(&mut self, num_tokens: usize) {
        self.adjust_kv_pages(-(num_tokens as isize));
    }

    /// Processes a batch of pending tokens to update the model's internal state.
    fn flush(&mut self) {
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

        let kv_ptrs: Vec<u32> = self.kv_pages.iter().map(|p| p.ptr()).collect();
        let p = self.queue.create_forward_pass();
        p.input_tokens(&pending_token_ids, &position_ids);
        p.kv_cache(&kv_ptrs, self.kv_page_last_len as u32);
        p.attention_mask(&mask);

        let _ = block_on(async move { p.execute().await });

        self.token_ids.extend(pending_token_ids);
        self.position_ids.extend(&position_ids);
    }

    /// Prepares and submits a decode-step forward pass without blocking.
    /// Returns the host result handle and the consumed pending state
    /// that must be committed after the result is ready.
    fn submit_decode_step(
        &mut self,
        sampler: &SamplerConfig,
    ) -> (HostForwardPassResult, Vec<u32>, Vec<u32>) {
        assert!(
            !self.token_ids_pending.is_empty(),
            "Must have at least one seed token"
        );

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

        if let Some(adapter_ptr) = self.adapter_ptr {
            p.set_adapter(adapter_ptr);

            if let Some(adapter_random_seed) = self.adapter_random_seed {
                p.set_adapter_seed(adapter_random_seed);
            }
        }

        let kv_ptrs: Vec<u32> = self.kv_pages.iter().map(|p| p.ptr()).collect();
        p.input_tokens(&pending_token_ids, &position_ids);
        p.kv_cache(&kv_ptrs, self.kv_page_last_len as u32);
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

        let host_result = p.submit().expect("Forward pass returned no result");
        (host_result, pending_token_ids, position_ids)
    }

    /// Commits the pending state from a decode step into the context.
    fn commit_decode_step(&mut self, pending_token_ids: &[u32], pending_position_ids: &[u32]) {
        self.token_ids.extend_from_slice(pending_token_ids);
        self.position_ids.extend_from_slice(pending_position_ids);
    }

    /// Performs a single, atomic autoregressive decoding step.
    ///
    /// Takes the pending tokens, runs a forward pass through the model, uses the
    /// provided sampler to choose the next token, and returns the sampled token ID.
    /// The pending tokens are consumed and moved to the main `token_ids` history,
    /// and the KV cache is updated accordingly.
    fn decode_step(&mut self, sampler: &SamplerConfig) -> u32 {
        let (host_result, pending_token_ids, position_ids) = self.submit_decode_step(sampler);

        let pollable = host_result.pollable();
        block_on(async {
            wstd::runtime::AsyncPollable::new(pollable).wait_for().await;
        });
        let sampled = host_result
            .get_tokens()
            .unwrap()
            .into_iter()
            .next()
            .unwrap();

        self.commit_decode_step(&pending_token_ids, &position_ids);

        sampled
    }

    /// Performs a single, atomic autoregressive decoding step and returns the
    /// full probability distribution over next tokens instead of a sampled token.
    /// This enables custom sampling logic (e.g. grammar-constrained decoding,
    /// watermarking) where the caller picks the token from the distribution.
    fn decode_step_dist(
        &mut self,
        temperature: f32,
        top_k: Option<u32>,
    ) -> (Vec<u32>, Vec<f32>) {
        assert!(
            !self.token_ids_pending.is_empty(),
            "Must have at least one seed token"
        );

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

        if let Some(adapter_ptr) = self.adapter_ptr {
            p.set_adapter(adapter_ptr);

            if let Some(adapter_random_seed) = self.adapter_random_seed {
                p.set_adapter_seed(adapter_random_seed);
            }
        }

        let kv_ptrs: Vec<u32> = self.kv_pages.iter().map(|p| p.ptr()).collect();
        p.input_tokens(&pending_token_ids, &position_ids);
        p.kv_cache(&kv_ptrs, self.kv_page_last_len as u32);
        p.attention_mask(&mask);

        let output_idx = pending_token_ids.len() as u32 - 1;
        p.output_distributions(&[output_idx], temperature, top_k);

        let res = block_on(async move { p.execute().await });
        let dist = res.distributions.unwrap().into_iter().next().unwrap();

        self.token_ids.extend(pending_token_ids);
        self.position_ids.extend(position_ids);

        (dist.ids, dist.probs)
    }

    /// Generates text autoregressively until a stop condition is met.
    ///
    /// Drives the text generation loop: in each iteration, calls `decode_step()` to
    /// sample the next token, adds it to the context, and checks the stop condition.
    fn generate(&mut self, sampler: &SamplerConfig, stop_config: &StopConfig) -> String {
        let mut generated_token_ids = Vec::new();

        loop {
            let next_token_id = self.decode_step(sampler);

            self.fill_token(next_token_id);

            generated_token_ids.push(next_token_id);

            let should_stop = generated_token_ids.len() >= stop_config.max_tokens as usize
                || stop_config
                    .eos_sequences
                    .iter()
                    .any(|seq| generated_token_ids.ends_with(seq));

            if should_stop {
                break;
            }
        }

        self.tokenizer.detokenize(&generated_token_ids)
    }

    /// Generates text using beam search decoding until a stop condition is met.
    ///
    /// Beam search explores multiple potential sequences simultaneously. At each step,
    /// it maintains `beam_size` candidate sequences, expands each with the top next
    /// tokens, scores them by cumulative log probability, and prunes to the top
    /// `beam_size` candidates. Upon completion, adopts the state of the winning beam.
    fn generate_with_beam(&mut self, stop_config: &StopConfig, beam_size: usize) -> String {
        let mut beams = Vec::new();
        beams.push((self.fork(), vec![], 0.0f32));

        loop {
            if let Some((_beam, generated_tokens, _)) = beams.iter().find(|(_, g, _)| {
                g.len() >= stop_config.max_tokens as usize
                    || stop_config.eos_sequences.iter().any(|seq| g.ends_with(seq))
            }) {
                let result = self.tokenizer.detokenize(generated_tokens);

                let winning_beam_idx = beams
                    .iter()
                    .position(|(_, g, _)| {
                        g.len() >= stop_config.max_tokens as usize
                            || stop_config.eos_sequences.iter().any(|seq| g.ends_with(seq))
                    })
                    .unwrap();
                let (beam, _, _) = &beams[winning_beam_idx];

                self.kv_page_last_len = beam.kv_page_last_len;
                self.token_ids = beam.token_ids.clone();
                self.token_ids_pending = beam.token_ids_pending.clone();
                self.kv_pages = beam.kv_pages.clone();

                return result;
            }

            let mut all_dists = Vec::new();
            for (beam, _, _) in beams.iter_mut() {
                let dist = beam.decode_step_dist(1.0, None);
                all_dists.push(dist);
            }

            let mut next_beams = Vec::new();
            for ((beam, generated, score), (ids, probs)) in beams.into_iter().zip(all_dists) {
                for i in 0..beam_size.min(ids.len()) {
                    let mut next_beam = beam.fork();
                    next_beam.fill_token(ids[i]);

                    let mut next_generated = generated.clone();
                    next_generated.push(ids[i]);

                    let next_score = score + probs[i].ln();

                    next_beams.push((next_beam, next_generated, next_score));
                }
            }

            next_beams.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(Ordering::Equal));
            next_beams.truncate(beam_size);
            beams = next_beams;
        }
    }

    /// Verifies speculative draft tokens against the model in a single batched
    /// forward pass with tree attention. Returns the greedy-verified accepted
    /// token sequence. Accepted tokens are automatically added to the context's
    /// pending buffer.
    ///
    /// Ported from the single-iteration core of `generate_with_drafter` in
    /// `inferlet/src/context.rs`.
    fn verify_draft(&mut self, draft_tokens: &[u32], draft_pos_ids: &[u32]) -> Vec<u32> {
        assert!(
            !self.token_ids_pending.is_empty(),
            "Must have at least one seed token"
        );

        let token_ids_pending = mem::take(&mut self.token_ids_pending);
        self.token_mask_pending.clear();

        let batch_tokens = [token_ids_pending.as_slice(), draft_tokens].concat();

        let pos_offset = self.position_ids.last().map(|&p| p + 1).unwrap_or(0);
        let pending_len = token_ids_pending.len() as u32;

        let batch_positions = {
            let mut positions = Vec::with_capacity(batch_tokens.len());
            positions.extend(pos_offset..pos_offset + pending_len);
            positions.extend(
                draft_pos_ids
                    .iter()
                    .map(|&pos| pos_offset + pending_len - 1 + pos),
            );
            positions
        };

        let batch_masks = {
            let mut masks = Vec::with_capacity(batch_tokens.len());

            let mut pending_token_brle = Brle::new(pos_offset as usize);

            for _ in 0..pending_len as usize {
                pending_token_brle.append(false);
                masks.push(pending_token_brle.get_buffer());
            }

            struct PredecessorInfo {
                draft_mask_idx: usize,
                pos: u32,
            }

            let mut draft_mask = vec![true; draft_tokens.len()];
            let mut predecessors: Vec<PredecessorInfo> = Vec::new();

            for (batch_idx, &pos) in batch_positions
                .iter()
                .enumerate()
                .skip(pending_len as usize)
            {
                let draft_mask_idx = batch_idx - pending_len as usize;

                while let Some(predecessor) = predecessors.last() {
                    if predecessor.pos != pos - 1 {
                        draft_mask[predecessor.draft_mask_idx] = true;
                        predecessors.pop();
                    } else {
                        break;
                    }
                }

                draft_mask[draft_mask_idx] = false;

                predecessors.push(PredecessorInfo {
                    draft_mask_idx,
                    pos,
                });

                let mut brle = pending_token_brle.clone();
                brle.extend(&Brle::from_bools(&draft_mask[..=draft_mask_idx]));
                masks.push(brle.get_buffer());
            }

            masks
        };

        self.grow_kv_pages(batch_tokens.len());

        let out_range = token_ids_pending.len() - 1..batch_tokens.len();

        let p = self.queue.create_forward_pass();

        if let Some(adapter_ptr) = self.adapter_ptr {
            p.set_adapter(adapter_ptr);
            if let Some(adapter_random_seed) = self.adapter_random_seed {
                p.set_adapter_seed(adapter_random_seed);
            }
        }

        let kv_ptrs: Vec<u32> = self.kv_pages.iter().map(|p| p.ptr()).collect();
        p.input_tokens(&batch_tokens, &batch_positions);
        p.kv_cache(&kv_ptrs, self.kv_page_last_len as u32);
        p.attention_mask(&batch_masks);
        p.output_distributions(
            &out_range.map(|x| x as u32).collect::<Vec<_>>(),
            0.0,
            None,
        );

        let res = block_on(async move { p.execute().await });
        let output_distributions = res.distributions.unwrap();

        let accepted_tokens = {
            let mut tokens = Vec::new();

            let first_dist = &output_distributions[0];
            let first_token = greedy_argmax(&first_dist.ids, &first_dist.probs);
            tokens.push(first_token);

            let mut draft_token_idx = 0;
            while draft_token_idx < draft_tokens.len() {
                let last_accepted = *tokens.last().unwrap();
                let draft_token = draft_tokens[draft_token_idx];

                if last_accepted == draft_token {
                    let next_dist = &output_distributions[draft_token_idx + 1];
                    let next_token = greedy_argmax(&next_dist.ids, &next_dist.probs);
                    tokens.push(next_token);

                    let has_child = draft_token_idx + 1 < draft_tokens.len()
                        && draft_pos_ids[draft_token_idx] + 1
                            == draft_pos_ids[draft_token_idx + 1];

                    if has_child {
                        draft_token_idx += 1;
                    } else {
                        break;
                    }
                } else {
                    let mut next_sibling_draft_idx = None;
                    let cur_depth = draft_pos_ids[draft_token_idx];

                    for idx in draft_token_idx + 1..draft_tokens.len() {
                        if draft_pos_ids[idx] < cur_depth {
                            break;
                        }
                        if draft_pos_ids[idx] == cur_depth {
                            next_sibling_draft_idx = Some(idx);
                            break;
                        }
                    }

                    if let Some(sibling_idx) = next_sibling_draft_idx {
                        draft_token_idx = sibling_idx;
                    } else {
                        break;
                    }
                }
            }

            tokens
        };

        self.shrink_kv_pages(draft_tokens.len());

        self.position_ids
            .extend(&batch_positions[..token_ids_pending.len()]);
        self.token_ids.extend(token_ids_pending.into_iter());

        self.fill_tokens(accepted_tokens.clone());

        accepted_tokens
    }

    /// Creates a safe, copy-on-write fork of the context.
    ///
    /// Creates a new context that shares the immutable history of the current one.
    /// Shared KV pages use reference counting so they are only deallocated when the
    /// last context holding a reference is dropped. If the last KV-cache page is not
    /// full, its tokens are moved to the `token_ids_pending` buffer of the new
    /// context to be recomputed, ensuring state isolation.
    fn fork(&self) -> Self {
        let (
            new_tokens,
            new_pending,
            new_kv_pages,
            new_kv_page_last_len,
            new_pos_ids,
            new_mask_pending,
        ) = if self.kv_page_last_len == self.kv_page_size
            && !self.token_ids_pending.is_empty()
        {
            // Easy case: the last page is full and there are pending tokens,
            // so we can share everything. When pending is empty we must fall
            // through to the hard case, which moves the last page's tokens
            // back into pending so that decode_step() can see seed tokens.
            (
                self.token_ids.clone(),
                self.token_ids_pending.clone(),
                self.kv_pages.clone(),
                self.kv_page_last_len,
                self.position_ids.clone(),
                self.token_mask_pending.clone(),
            )
        } else {
            let kept_kv_page_len = self.kv_pages.len().saturating_sub(1);
            let kept_tokens_len = kept_kv_page_len * self.kv_page_size;

            let forked_token_ids = self.token_ids[..kept_tokens_len].to_vec();
            let forked_kv_pages = self.kv_pages[..kept_kv_page_len].to_vec();
            let forked_pos_ids = self.position_ids[..kept_tokens_len].to_vec();

            let forked_pending_token_ids = [
                &self.token_ids[kept_tokens_len..],
                &self.token_ids_pending[..],
            ]
            .concat();

            let forked_last_kv_page_len = if !forked_kv_pages.is_empty() {
                self.kv_page_size
            } else {
                0
            };

            let mut mask_builder = self.token_mask_current.clone();
            let parent_total_mask_len = self.token_ids.len() + self.token_ids_pending.len();
            mask_builder.remove_range(kept_tokens_len, parent_total_mask_len);

            let mut forked_mask_pending = Vec::with_capacity(forked_pending_token_ids.len());
            for _ in 0..forked_pending_token_ids.len() {
                mask_builder.append(false);
                forked_mask_pending.push(mask_builder.clone());
            }

            (
                forked_token_ids,
                forked_pending_token_ids,
                forked_kv_pages,
                forked_last_kv_page_len,
                forked_pos_ids,
                forked_mask_pending,
            )
        };

        let model_name = self.model.get_name();
        let host_model = get_model(&model_name).expect("Failed to get model");
        let queue = Queue::from_host_model(&host_model);
        let prompt_template = self.model.get_prompt_template();
        let tokenizer = self.model.get_tokenizer();
        let formatter =
            ChatFormatter::new(prompt_template).expect("Failed to create chat formatter");

        Context {
            model: self.model.clone(),
            queue,
            tokenizer,
            formatter,
            token_ids: new_tokens,
            token_ids_pending: new_pending,
            token_mask_pending: new_mask_pending,
            token_mask_current: self.token_mask_current.clone(),
            position_ids: new_pos_ids,
            kv_pages: new_kv_pages,
            kv_page_last_len: new_kv_page_last_len,
            kv_page_size: self.kv_page_size,
            adapter_ptr: self.adapter_ptr,
            adapter_random_seed: self.adapter_random_seed,
            begin_of_sequence: self.begin_of_sequence,
        }
    }
}

// No custom Drop needed: KvPage handles deallocation via Rc reference counting.
// When a Context is dropped, its Vec<KvPage> is dropped, which drops each KvPage,
// and each KvPage only deallocates the underlying page when the last reference is gone.

pub(crate) struct ContextImpl {
    inner: Rc<RefCell<Context>>,
}

impl GuestContext for ContextImpl {
    fn new(wit_model: crate::exports::inferlib::inference::models::ModelBorrow<'_>) -> Self {
        let model_impl: &ModelImpl = wit_model.get();
        let model = model_impl.inner.borrow().clone();
        let inner = Context::new(&model);
        ContextImpl {
            inner: Rc::new(RefCell::new(inner)),
        }
    }

    fn from_imported_state(
        wit_model: crate::exports::inferlib::inference::models::ModelBorrow<'_>,
        kv_page_ptrs: Vec<u32>,
        prefix_tokens: Vec<u32>,
        kv_page_last_len: u32,
    ) -> crate::exports::inferlib::inference::inference::Context {
        let model_impl: &ModelImpl = wit_model.get();
        let model = model_impl.inner.borrow().clone();
        let inner = Context::from_imported_state(
            &model,
            kv_page_ptrs,
            prefix_tokens,
            kv_page_last_len as usize,
        );
        crate::exports::inferlib::inference::inference::Context::new(ContextImpl {
            inner: Rc::new(RefCell::new(inner)),
        })
    }

    fn fill(&self, text: String) {
        self.inner.borrow_mut().fill(&text);
    }

    fn fill_tokens(&self, token_ids: Vec<u32>) {
        self.inner.borrow_mut().fill_tokens(token_ids);
    }

    fn fill_token(&self, token_id: u32) {
        self.inner.borrow_mut().fill_token(token_id);
    }

    fn fill_system(&self, text: String) {
        self.inner.borrow_mut().fill_system(&text);
    }

    fn fill_user(&self, text: String) {
        self.inner.borrow_mut().fill_user(&text);
    }

    fn fill_user_only(&self, text: String) {
        self.inner.borrow_mut().fill_user_only(&text);
    }

    fn fill_assistant(&self, text: String) {
        self.inner.borrow_mut().fill_assistant(&text);
    }

    fn mask_tokens(&self, indices: Vec<u32>, mask: bool) {
        let indices: Vec<usize> = indices.into_iter().map(|i| i as usize).collect();
        self.inner.borrow_mut().mask_tokens(&indices, mask);
    }

    fn mask_token_range(&self, start: u32, end: u32, mask: bool) {
        self.inner
            .borrow_mut()
            .mask_token_range(start as usize, end as usize, mask);
    }

    fn mask_token(&self, index: u32, mask: bool) {
        self.inner.borrow_mut().mask_token(index as usize, mask);
    }

    fn drop_masked_kv_pages(&self) {
        self.inner.borrow_mut().drop_masked_kv_pages();
    }

    fn set_adapter(&self, adapter_ptr: u32) {
        self.inner.borrow_mut().set_adapter(adapter_ptr);
    }

    fn remove_adapter(&self) {
        self.inner.borrow_mut().remove_adapter();
    }

    fn set_adapter_random_seed(&self, seed: i64) {
        self.inner.borrow_mut().set_adapter_random_seed(seed);
    }

    fn flush(&self) {
        self.inner.borrow_mut().flush();
    }

    fn decode_step(&self, sampler_config: SamplerConfig) -> u32 {
        self.inner.borrow_mut().decode_step(&sampler_config)
    }

    fn decode_step_dist(
        &self,
        temperature: f32,
        top_k: Option<u32>,
    ) -> crate::exports::inferlib::inference::queues::Distribution {
        let (ids, probs) = self
            .inner
            .borrow_mut()
            .decode_step_dist(temperature, top_k);
        crate::exports::inferlib::inference::queues::Distribution { ids, probs }
    }

    fn generate(&self, sampler_config: SamplerConfig, stop_config: StopConfig) -> String {
        self.inner
            .borrow_mut()
            .generate(&sampler_config, &stop_config)
    }

    fn generate_with_beam(&self, stop_config: StopConfig, beam_size: u32) -> String {
        self.inner
            .borrow_mut()
            .generate_with_beam(&stop_config, beam_size as usize)
    }

    fn verify_draft(&self, draft_tokens: Vec<u32>, draft_pos_ids: Vec<u32>) -> Vec<u32> {
        self.inner
            .borrow_mut()
            .verify_draft(&draft_tokens, &draft_pos_ids)
    }

    fn fork(&self) -> crate::exports::inferlib::inference::inference::Context {
        let forked = self.inner.borrow().fork();
        crate::exports::inferlib::inference::inference::Context::new(ContextImpl {
            inner: Rc::new(RefCell::new(forked)),
        })
    }

    fn get_text(&self) -> String {
        self.inner.borrow().get_text()
    }

    fn get_token_ids(&self) -> Vec<u32> {
        self.inner.borrow().get_token_ids().to_vec()
    }

    fn get_kv_page_ptrs(&self) -> Vec<u32> {
        self.inner.borrow().get_kv_page_ptrs()
    }

    fn get_kv_page_last_len(&self) -> u32 {
        self.inner.borrow().get_kv_page_last_len() as u32
    }

    fn decode_step_async(
        &self,
        sampler: SamplerConfig,
    ) -> crate::exports::inferlib::inference::inference::DecodeStepFuture {
        let (host_result, pending_token_ids, position_ids) =
            self.inner.borrow_mut().submit_decode_step(&sampler);

        crate::exports::inferlib::inference::inference::DecodeStepFuture::new(
            DecodeStepFutureImpl {
                context: Rc::clone(&self.inner),
                host_result: RefCell::new(Some(host_result)),
                pending_token_ids,
                pending_position_ids: position_ids,
            },
        )
    }

    fn flush_async(&self) -> Option<crate::exports::inferlib::inference::inference::FlushFuture> {
        let mut ctx = self.inner.borrow_mut();

        if ctx.token_ids_pending.is_empty() {
            return None;
        }

        let process_count = ctx.token_ids_pending.len();

        let pending_token_ids = ctx
            .token_ids_pending
            .drain(..process_count)
            .collect::<Vec<u32>>();

        let mask = ctx
            .token_mask_pending
            .drain(..process_count)
            .map(|b| b.get_buffer())
            .collect::<Vec<Vec<u32>>>();

        let last_pos = ctx.position_ids.last().map(|&p| p + 1).unwrap_or(0);
        let position_ids =
            (last_pos..(last_pos + pending_token_ids.len() as u32)).collect::<Vec<u32>>();

        ctx.grow_kv_pages(pending_token_ids.len());

        let kv_ptrs: Vec<u32> = ctx.kv_pages.iter().map(|p| p.ptr()).collect();
        let p = ctx.queue.create_forward_pass();
        p.input_tokens(&pending_token_ids, &position_ids);
        p.kv_cache(&kv_ptrs, ctx.kv_page_last_len as u32);
        p.attention_mask(&mask);

        let host_result = p.submit();

        ctx.token_ids.extend(&pending_token_ids);
        ctx.position_ids.extend(&position_ids);

        Some(
            crate::exports::inferlib::inference::inference::FlushFuture::new(FlushFutureImpl {
                host_result: RefCell::new(host_result),
            }),
        )
    }

    fn generate_async(
        &self,
        sampler: SamplerConfig,
        stop_config: StopConfig,
    ) -> crate::exports::inferlib::inference::inference::GenerateFuture {
        crate::exports::inferlib::inference::inference::GenerateFuture::new(GenerateFutureImpl {
            context: Rc::clone(&self.inner),
            sampler,
            stop_config,
            state: RefCell::new(GenerateFutureState {
                generated_token_ids: Vec::new(),
                phase: GeneratePhase::Ready,
            }),
        })
    }
}

pub(crate) struct DecodeStepFutureImpl {
    context: Rc<RefCell<Context>>,
    host_result: RefCell<Option<HostForwardPassResult>>,
    pending_token_ids: Vec<u32>,
    pending_position_ids: Vec<u32>,
}

impl GuestDecodeStepFuture for DecodeStepFutureImpl {
    fn pollable(&self) -> wasip2::io::poll::Pollable {
        self.host_result
            .borrow()
            .as_ref()
            .expect("Forward pass result already consumed")
            .pollable()
    }

    fn get(&self) -> Option<u32> {
        let host_result = self.host_result.borrow();
        let result = host_result.as_ref()?;
        let tokens = result.get_tokens()?;
        let sampled = tokens.into_iter().next()?;

        let mut ctx = self.context.borrow_mut();
        ctx.commit_decode_step(&self.pending_token_ids, &self.pending_position_ids);

        Some(sampled)
    }
}

pub(crate) struct FlushFutureImpl {
    host_result: RefCell<Option<HostForwardPassResult>>,
}

impl GuestFlushFuture for FlushFutureImpl {
    fn pollable(&self) -> wasip2::io::poll::Pollable {
        self.host_result
            .borrow()
            .as_ref()
            .expect("Flush future has no pending result")
            .pollable()
    }

    fn is_ready(&self) -> bool {
        self.host_result
            .borrow()
            .as_ref()
            .map(|r| r.pollable().ready())
            .unwrap_or(true)
    }
}

enum GeneratePhase {
    Ready,
    Pending {
        host_result: HostForwardPassResult,
        pending_token_ids: Vec<u32>,
        pending_position_ids: Vec<u32>,
    },
    Done,
}

struct GenerateFutureState {
    generated_token_ids: Vec<u32>,
    phase: GeneratePhase,
}

pub(crate) struct GenerateFutureImpl {
    context: Rc<RefCell<Context>>,
    sampler: SamplerConfig,
    stop_config: StopConfig,
    state: RefCell<GenerateFutureState>,
}

impl GuestGenerateFuture for GenerateFutureImpl {
    fn pollable(&self) -> wasip2::io::poll::Pollable {
        let mut state = self.state.borrow_mut();
        match &state.phase {
            GeneratePhase::Ready => {
                let (host_result, pending_token_ids, pending_position_ids) =
                    self.context.borrow_mut().submit_decode_step(&self.sampler);
                let pollable = host_result.pollable();
                state.phase = GeneratePhase::Pending {
                    host_result,
                    pending_token_ids,
                    pending_position_ids,
                };
                pollable
            }
            GeneratePhase::Pending { host_result, .. } => host_result.pollable(),
            GeneratePhase::Done => {
                panic!("pollable() called on completed generate future")
            }
        }
    }

    fn get(&self) -> Option<String> {
        let mut state = self.state.borrow_mut();

        let (host_result, pending_token_ids, pending_position_ids) =
            match mem::replace(&mut state.phase, GeneratePhase::Ready) {
                GeneratePhase::Pending {
                    host_result,
                    pending_token_ids,
                    pending_position_ids,
                } => (host_result, pending_token_ids, pending_position_ids),
                other => {
                    state.phase = other;
                    return None;
                }
            };

        let token = host_result
            .get_tokens()
            .and_then(|t| t.into_iter().next())
            .expect("Decode step produced no token");

        {
            let mut ctx = self.context.borrow_mut();
            ctx.commit_decode_step(&pending_token_ids, &pending_position_ids);
            ctx.fill_token(token);
        }

        state.generated_token_ids.push(token);

        let should_stop = state.generated_token_ids.len() >= self.stop_config.max_tokens as usize
            || self
                .stop_config
                .eos_sequences
                .iter()
                .any(|seq| state.generated_token_ids.ends_with(seq));

        if should_stop {
            let ctx = self.context.borrow();
            let result = ctx.tokenizer.detokenize(&state.generated_token_ids);
            state.phase = GeneratePhase::Done;
            Some(result)
        } else {
            None
        }
    }
}
