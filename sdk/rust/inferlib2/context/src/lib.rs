mod brle;
mod chat;

wit_bindgen::generate!({
    path: "wit",
    world: "context-provider",
    with: {
        "inferlib2:engine/models": inferlib2_engine_bindings::engine::models,
        "inferlib2:engine/queues": inferlib2_engine_bindings::engine::queues,
    },
});

use exports::inferlib2::context::inference::{
    Guest as InferenceGuest, GuestContext, SamplerConfig, StopConfig,
};
use exports::inferlib2::context::formatter::{
    Guest as FormatterGuest, GuestChatFormatter, ToolCall as WitToolCall,
};

use inferlib2_engine_bindings::{Model as WitModel, Tokenizer, Queue as WitQueue};

use brle::Brle;
use chat::ChatFormatter;

use std::cell::RefCell;
use std::cmp::Ordering;
use std::mem;
use std::rc::Rc;

// ── Inference (Context) implementation ──

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
    fn new(wit_model: &WitModel) -> Self {
        let model_name = wit_model.get_name();
        let queue = WitQueue::from_model_name(&model_name);
        let kv_page_size = wit_model.get_kv_page_size() as usize;
        let prompt_template = wit_model.get_prompt_template();
        let tokenizer = wit_model.get_tokenizer();
        let formatter =
            ChatFormatter::new(prompt_template).expect("Failed to create chat formatter");

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

    fn get_token_ids(&self) -> &[u32] {
        &self.token_ids
    }

    fn get_text(&self) -> String {
        self.tokenizer.detokenize(&self.token_ids)
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

    fn fill_assistant(&mut self, text: &str) {
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

    fn grow_kv_pages(&mut self, num_tokens: usize) {
        self.adjust_kv_pages(num_tokens as isize);
    }

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

        let p = self.queue.create_forward_pass();
        p.input_tokens(&pending_token_ids, &position_ids);
        p.kv_cache(&self.kv_page_ptrs, self.kv_page_last_len as u32);
        p.attention_mask(&mask);

        let _ = p.execute();

        self.token_ids.extend(pending_token_ids);
        self.position_ids.extend(&position_ids);
    }

    fn decode_step(&mut self, sampler: &SamplerConfig) -> u32 {
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

        self.token_ids.extend(pending_token_ids);
        self.position_ids.extend(position_ids);

        sampled
    }

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
}

impl Drop for Context {
    fn drop(&mut self) {
        if !self.kv_page_ptrs.is_empty() {
            self.queue.deallocate_kv_pages(&self.kv_page_ptrs);
        }
    }
}

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

// ── Chat Formatter WIT export implementation ──

struct ChatFormatterImpl {
    formatter: RefCell<ChatFormatter>,
}

impl GuestChatFormatter for ChatFormatterImpl {
    fn new(template: String) -> Result<Self, String> {
        let formatter = ChatFormatter::new(template)?;
        Ok(ChatFormatterImpl {
            formatter: RefCell::new(formatter),
        })
    }

    fn add_system(&self, content: String) {
        self.formatter.borrow_mut().add_system(content);
    }

    fn add_user(&self, content: String) {
        self.formatter.borrow_mut().add_user(content);
    }

    fn add_assistant(&self, content: String) {
        self.formatter.borrow_mut().add_assistant(content);
    }

    fn add_assistant_response(
        &self,
        content: String,
        reasoning: Option<String>,
        tool_calls: Option<Vec<WitToolCall>>,
    ) {
        let internal_tool_calls = tool_calls.map(|calls| {
            calls
                .into_iter()
                .map(|tc| {
                    let args: serde_json::Value =
                        serde_json::from_str(&tc.arguments).unwrap_or(serde_json::Value::String(tc.arguments));
                    chat::ToolCall {
                        name: tc.name,
                        arguments: args,
                    }
                })
                .collect()
        });

        self.formatter
            .borrow_mut()
            .add_assistant_response(content, reasoning, internal_tool_calls);
    }

    fn add_tool(&self, content: String) {
        self.formatter.borrow_mut().add_tool(content);
    }

    fn has_messages(&self) -> bool {
        self.formatter.borrow().has_messages()
    }

    fn clear(&self) {
        self.formatter.borrow_mut().clear();
    }

    fn render(&self, add_generation_prompt: bool, begin_of_sequence: bool) -> String {
        self.formatter
            .borrow()
            .render(add_generation_prompt, begin_of_sequence)
    }
}

// ── Combined export ──

struct ContextComponentImpl;

impl InferenceGuest for ContextComponentImpl {
    type Context = ContextImpl;
}

impl FormatterGuest for ContextComponentImpl {
    type ChatFormatter = ChatFormatterImpl;
}

export!(ContextComponentImpl);
