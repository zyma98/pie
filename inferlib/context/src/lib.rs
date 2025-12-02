// Generate WIT bindings for exports
wit_bindgen::generate!({
    path: "wit",
    world: "context-provider",
    generate_all,
});

use exports::inferlib::context::inference::{Guest, GuestContext, SamplerConfig, StopConfig};

// Import the WIT model type for the constructor
use inferlib::model::models::Model as WitModel;

// Import types from the legacy library that Context depends on
use inferlet::brle::Brle;
use inferlet::forward::{Forward, KvPage};
use inferlet::stop_condition::{ends_with_any, max_len, StopCondition};
use inferlet::{api, ChatFormatter, Queue, Sampler, Tokenizer};

use std::cell::RefCell;
use std::cmp::Ordering;
use std::mem;
use std::rc::Rc;

struct InferenceImpl;

impl Guest for InferenceImpl {
    type Context = ContextImpl;
}

/// Internal Model struct that wraps the host API Model
struct Model {
    inner: Rc<api::Model>,
}

impl Model {
    /// Create from the host API model
    fn from_api(inner: api::Model) -> Self {
        Model {
            inner: Rc::new(inner),
        }
    }

    /// Get the prompt template
    pub fn get_prompt_template(&self) -> String {
        self.inner.get_prompt_template()
    }

    /// Get the KV page size
    pub fn get_kv_page_size(&self) -> u32 {
        self.inner.get_kv_page_size()
    }

    /// Create a queue for this model using the legacy Queue
    pub fn create_queue(&self) -> Queue {
        // Use the legacy library's Model to create the queue
        // This requires getting a legacy Model first
        let model_name = self.inner.get_name();
        let legacy_model = inferlet::get_model(&model_name).expect("Failed to get legacy model");
        legacy_model.create_queue()
    }

    /// Get a tokenizer for this model using the legacy Tokenizer
    pub fn get_tokenizer(&self) -> Tokenizer {
        let model_name = self.inner.get_name();
        let legacy_model = inferlet::get_model(&model_name).expect("Failed to get legacy model");
        Tokenizer::new(&legacy_model)
    }
}

impl Clone for Model {
    fn clone(&self) -> Self {
        Model {
            inner: Rc::clone(&self.inner),
        }
    }
}

/// Internal Context struct that re-implements the legacy Context
struct Context {
    queue: Queue,
    model: Model,
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

    begin_of_sequence: bool,
}

impl Context {
    pub fn new(model: &Model) -> Self {
        let queue = model.create_queue();
        let kv_page_size = model.get_kv_page_size() as usize;
        let tokenizer = model.get_tokenizer();
        Context {
            queue,
            model: model.clone(),
            tokenizer,
            formatter: ChatFormatter::new(),
            token_ids: Vec::new(),
            token_ids_pending: Vec::new(),
            token_mask_pending: Vec::new(),
            token_mask_current: Brle::new(0),
            position_ids: Vec::new(),
            kv_pages: Vec::new(),
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
            // always fill with false - we don't mask tokens to mask itself.
            self.token_mask_current.append(false);
            self.token_mask_pending
                .push(self.token_mask_current.clone())
        }
        self.begin_of_sequence = false;
    }

    pub fn fill_token(&mut self, new_token_id: u32) {
        self.token_ids_pending.push(new_token_id);
        self.token_mask_current.append(false);
        self.token_mask_pending
            .push(self.token_mask_current.clone());
        self.begin_of_sequence = false;
    }

    pub fn fill_system(&mut self, text: &str) {
        self.formatter.system(text);
        self.flush_chat_messages2(false);
    }

    pub fn fill_user(&mut self, text: &str) {
        self.formatter.user(text);
        self.flush_chat_messages2(true);
    }

    pub fn fill_assistant(&mut self, text: &str) {
        self.formatter.assistant(text);
        self.flush_chat_messages2(false);
    }

    fn flush_chat_messages2(&mut self, add_generation_prompt: bool) {
        if self.formatter.has_messages() {
            let p = self.formatter.render(
                &self.model.get_prompt_template(),
                add_generation_prompt,
                self.begin_of_sequence,
            );
            self.begin_of_sequence = false;
            self.formatter.clear();
            self.fill(&p);
        }
    }

    /// Adjusts the number of KV pages to match the required number of tokens.
    fn adjust_kv_pages(&mut self, num_tokens: isize) {
        if num_tokens == 0 {
            return;
        }

        let current_tokens = if self.kv_pages.is_empty() {
            self.kv_page_last_len
        } else {
            (self.kv_pages.len() - 1) * self.kv_page_size + self.kv_page_last_len
        };

        // Safely calculate the new total number of tokens after the adjustment.
        let new_total_tokens = match current_tokens.checked_add_signed(num_tokens) {
            Some(n) => n,
            None => panic!("Token count adjustment resulted in underflow"),
        };

        let current_pages = self.kv_pages.len();
        let required_pages = new_total_tokens.div_ceil(self.kv_page_size);

        match required_pages.cmp(&current_pages) {
            Ordering::Greater => {
                // Grow: Allocate new pages if more are needed.
                let new_pages_needed = required_pages - current_pages;
                let new_kv_page_ids = self.queue.new_kv_pages(new_pages_needed);
                self.kv_pages.extend(new_kv_page_ids);
            }
            Ordering::Less => {
                // Shrink: Deallocate pages that are no longer needed.
                let _ = self.kv_pages.split_off(required_pages);
            }
            Ordering::Equal => {
                // No change in the number of pages is required.
            }
        }

        // Finally, update the length of the last page based on the new total.
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

    /// Processes a batch of pending tokens to update the model's internal state.
    pub async fn flush(&mut self) {
        if self.token_ids_pending.is_empty() {
            return;
        }
        let process_count = self.token_ids_pending.len();

        // Process all pending tokens
        let pending_token_ids = self
            .token_ids_pending
            .drain(..process_count)
            .collect::<Vec<u32>>();

        let mask = self
            .token_mask_pending
            .drain(..process_count)
            .map(|b| b.buffer)
            .collect::<Vec<Vec<u32>>>();

        let last_pos = self.position_ids.last().map(|&p| p + 1).unwrap_or(0);
        let position_ids =
            (last_pos..(last_pos + pending_token_ids.len() as u32)).collect::<Vec<u32>>();

        self.grow_kv_pages(pending_token_ids.len());

        let p = self.queue.create_forward_pass();
        p.input_tokens(&pending_token_ids, &position_ids);
        p.kv_cache(&self.kv_pages, self.kv_page_last_len);
        p.attention_mask(&mask);

        let _ = p.execute().await;

        self.token_ids.extend(pending_token_ids);
        self.position_ids.extend(&position_ids);
    }

    /// Performs a single, atomic autoregressive decoding step.
    pub async fn decode_step(&mut self, sampler: &Sampler) -> u32 {
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
            .map(|brie| brie.buffer)
            .collect::<Vec<Vec<u32>>>();

        let p = self.queue.create_forward_pass();

        p.input_tokens(&pending_token_ids, &position_ids);
        p.kv_cache(&self.kv_pages, self.kv_page_last_len);
        p.attention_mask(&mask);

        let output_idx = pending_token_ids.len() as u32 - 1;
        match sampler {
            Sampler::Custom {
                temperature,
                sampler: _sampler,
            } => {
                p.output_distributions(&[output_idx], *temperature, None);
            }
            Sampler::Multinomial { temperature } => {
                p.output_tokens(&[output_idx], *temperature);
            }
            Sampler::TopP { temperature, top_p } => {
                p.output_tokens_top_p(&[output_idx], *temperature, *top_p);
            }
            Sampler::TopK { temperature, top_k } => {
                p.output_tokens_top_k(&[output_idx], *temperature, *top_k);
            }
            Sampler::MinP { temperature, min_p } => {
                p.output_tokens_min_p(&[output_idx], *temperature, *min_p);
            }
            Sampler::TopKTopP {
                temperature,
                top_k,
                top_p,
            } => {
                p.output_tokens_top_k_top_p(&[output_idx], *temperature, *top_k, *top_p);
            }
        }

        let res = p.execute().await;

        let sampled = match sampler {
            Sampler::Custom {
                temperature: _temperature,
                sampler,
            } => {
                let dist = res.distributions.unwrap().into_iter().next().unwrap();
                sampler.sample(&dist.ids, &dist.probs)
            }
            _ => res.tokens.unwrap().into_iter().next().unwrap(),
        };

        self.token_ids.extend(pending_token_ids);
        self.position_ids.extend(position_ids);

        sampled
    }

    /// Generates text autoregressively until a stop condition is met.
    pub async fn generate<S: StopCondition>(
        &mut self,
        sampler: Sampler,
        stop_condition: S,
    ) -> String {
        let mut generated_token_ids = Vec::new();

        // The autoregressive generation loop
        loop {
            let next_token_id = self.decode_step(&sampler).await;

            self.fill_token(next_token_id);

            generated_token_ids.push(next_token_id);

            if stop_condition.check(&generated_token_ids) {
                break;
            }
        }

        self.tokenizer.detokenize(&generated_token_ids)
    }
}

// WIT interface wrapper
struct ContextImpl {
    inner: Rc<RefCell<Context>>,
}

impl GuestContext for ContextImpl {
    fn new(wit_model: &WitModel) -> Self {
        // Get the model name from the WIT model and look up the host model
        let model_name = wit_model.get_name();
        let api_model = api::runtime::get_model(&model_name).expect("Failed to get model by name");
        let model = Model::from_api(api_model);

        let inner = Context::new(&model);
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
        // Convert WIT sampler config to inferlet Sampler
        let sampler = match sampler_config {
            SamplerConfig::Greedy => Sampler::greedy(),
            SamplerConfig::Multinomial(temp) => Sampler::Multinomial { temperature: temp },
            SamplerConfig::TopP((temp, top_p)) => Sampler::top_p(temp, top_p),
            SamplerConfig::TopK((temp, top_k)) => Sampler::top_k(temp, top_k),
            SamplerConfig::MinP((temp, min_p)) => Sampler::min_p(temp, min_p),
            SamplerConfig::TopKTopP((temp, top_k, top_p)) => {
                Sampler::top_k_top_p(temp, top_k, top_p)
            }
        };

        // Convert WIT stop config to inferlet stop condition
        let stop_cond =
            max_len(stop_config.max_tokens as usize).or(ends_with_any(stop_config.eos_sequences));

        // Clone the Rc to move into the async block
        let inner = Rc::clone(&self.inner);

        // Run the async generate in a blocking context
        inferlet::wstd::runtime::block_on(async move {
            inner.borrow_mut().generate(sampler, stop_cond).await
        })
    }

    fn flush(&self) {
        // Clone the Rc to move into the async block
        let inner = Rc::clone(&self.inner);

        inferlet::wstd::runtime::block_on(async move {
            inner.borrow_mut().flush().await;
        });
    }

    fn get_text(&self) -> String {
        self.inner.borrow().get_text()
    }

    fn get_token_ids(&self) -> Vec<u32> {
        self.inner.borrow().get_token_ids().to_vec()
    }
}

export!(InferenceImpl);
