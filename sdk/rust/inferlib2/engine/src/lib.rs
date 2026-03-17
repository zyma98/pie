wit_bindgen::generate!({
    path: "wit",
    world: "engine-provider",
    generate_all,
    with: {
        "wasi:io/poll@0.2.4": wasip2::io::poll,
    },
});

// ── Models implementation ──

use exports::inferlib2::engine::models::{
    Guest as ModelsGuest, GuestModel, GuestTokenizer,
};

use crate::inferlet::core::common::Model as HostModel;
use crate::inferlet::core::runtime::{get_all_models, get_model};
use crate::inferlet::core::tokenize::{Tokenizer as HostTokenizer, get_tokenizer};

use std::cell::RefCell;
use std::collections::HashSet;
use std::rc::Rc;

struct Model {
    inner: Rc<HostModel>,
}

impl Model {
    fn from_host(inner: HostModel) -> Self {
        Model {
            inner: Rc::new(inner),
        }
    }

    fn get_by_name(name: &str) -> Option<Self> {
        get_model(name).map(|inner| Model::from_host(inner))
    }

    fn get_auto() -> Self {
        let models = get_all_models();
        if models.is_empty() {
            panic!("No models available");
        }
        let model_name = &models[0];
        get_model(model_name)
            .map(|inner| Model::from_host(inner))
            .expect("Failed to get first model")
    }

    fn get_all_names() -> Vec<String> {
        get_all_models()
    }

    fn get_name(&self) -> String {
        self.inner.get_name()
    }

    fn get_traits(&self) -> Vec<String> {
        self.inner.get_traits()
    }

    fn has_traits(&self, required_traits: &[&str]) -> bool {
        let available_traits_vec = self.get_traits();
        let available_traits: HashSet<&str> =
            available_traits_vec.iter().map(String::as_str).collect();
        required_traits
            .iter()
            .all(|t| available_traits.contains(t))
    }

    fn get_description(&self) -> String {
        self.inner.get_description()
    }

    fn get_prompt_template(&self) -> String {
        self.inner.get_prompt_template()
    }

    fn eos_tokens(&self) -> Vec<Vec<u32>> {
        let tokenizer = get_tokenizer(&self.inner);
        self.inner
            .get_stop_tokens()
            .into_iter()
            .map(|t| tokenizer.tokenize(&t))
            .collect()
    }

    fn get_service_id(&self) -> u32 {
        self.inner.get_service_id()
    }

    fn get_kv_page_size(&self) -> u32 {
        self.inner.get_kv_page_size()
    }

    fn get_tokenizer(&self) -> Tokenizer {
        let host_tokenizer = get_tokenizer(&self.inner);
        Tokenizer {
            inner: Rc::new(host_tokenizer),
        }
    }
}

impl Clone for Model {
    fn clone(&self) -> Self {
        Model {
            inner: Rc::clone(&self.inner),
        }
    }
}

struct Tokenizer {
    inner: Rc<HostTokenizer>,
}

impl Tokenizer {
    fn tokenize(&self, text: &str) -> Vec<u32> {
        self.inner.tokenize(text)
    }

    fn detokenize(&self, tokens: &[u32]) -> String {
        self.inner.detokenize(tokens)
    }

    fn get_vocabs(&self) -> (Vec<u32>, Vec<Vec<u8>>) {
        self.inner.get_vocabs()
    }
}

struct ModelImpl {
    inner: RefCell<Model>,
}

impl GuestModel for ModelImpl {
    fn get_by_name(name: String) -> Option<exports::inferlib2::engine::models::Model> {
        Model::get_by_name(&name).map(|model| {
            exports::inferlib2::engine::models::Model::new(ModelImpl {
                inner: RefCell::new(model),
            })
        })
    }

    fn get_auto() -> exports::inferlib2::engine::models::Model {
        exports::inferlib2::engine::models::Model::new(ModelImpl {
            inner: RefCell::new(Model::get_auto()),
        })
    }

    fn get_all_names() -> Vec<String> {
        Model::get_all_names()
    }

    fn get_name(&self) -> String {
        self.inner.borrow().get_name()
    }

    fn get_traits(&self) -> Vec<String> {
        self.inner.borrow().get_traits()
    }

    fn has_traits(&self, required_traits: Vec<String>) -> bool {
        let traits_refs: Vec<&str> = required_traits.iter().map(|s| s.as_str()).collect();
        self.inner.borrow().has_traits(&traits_refs)
    }

    fn get_description(&self) -> String {
        self.inner.borrow().get_description()
    }

    fn get_prompt_template(&self) -> String {
        self.inner.borrow().get_prompt_template()
    }

    fn eos_tokens(&self) -> Vec<Vec<u32>> {
        self.inner.borrow().eos_tokens()
    }

    fn get_service_id(&self) -> u32 {
        self.inner.borrow().get_service_id()
    }

    fn get_kv_page_size(&self) -> u32 {
        self.inner.borrow().get_kv_page_size()
    }

    fn get_tokenizer(&self) -> exports::inferlib2::engine::models::Tokenizer {
        let tokenizer = self.inner.borrow().get_tokenizer();
        exports::inferlib2::engine::models::Tokenizer::new(TokenizerImpl {
            inner: RefCell::new(tokenizer),
        })
    }
}

struct TokenizerImpl {
    inner: RefCell<Tokenizer>,
}

impl GuestTokenizer for TokenizerImpl {
    fn tokenize(&self, text: String) -> Vec<u32> {
        self.inner.borrow().tokenize(&text)
    }

    fn detokenize(&self, tokens: Vec<u32>) -> String {
        self.inner.borrow().detokenize(&tokens)
    }

    fn get_vocabs(&self) -> (Vec<u32>, Vec<Vec<u8>>) {
        self.inner.borrow().get_vocabs()
    }
}

// ── Queues implementation ──

use exports::inferlib2::engine::queues::{
    Distribution, ForwardPassResult, Guest as QueuesGuest, GuestForwardPass, GuestQueue, Priority,
    ResourceType,
};

use crate::inferlet::core::common::{
    Queue as HostQueue, allocate_resources, deallocate_resources,
    export_resources, get_all_exported_resources, import_resources, release_exported_resources,
};
use crate::inferlet::core::forward::{
    ForwardPass as HostForwardPass, attention_mask, create_forward_pass, input_embeddings,
    input_tokens, kv_cache, output_distributions, output_embeddings, output_tokens,
    output_tokens_min_p, output_tokens_top_k, output_tokens_top_k_top_p, output_tokens_top_p,
};

use crate::inferlet::adapter::common::set_adapter;
use crate::inferlet::zo::evolve::set_adapter_seed;

use wstd::runtime::{AsyncPollable, block_on};

struct Queue {
    inner: Rc<HostQueue>,
    service_id: u32,
}

impl Queue {
    fn from_host_model(model: &HostModel) -> Self {
        let queue = model.create_queue();
        let service_id = model.get_service_id();
        Queue {
            inner: Rc::new(queue),
            service_id,
        }
    }

    fn get_service_id(&self) -> u32 {
        self.service_id
    }

    async fn synchronize(&self) -> bool {
        let future = self.inner.synchronize();
        let pollable = future.pollable();
        AsyncPollable::new(pollable).wait_for().await;
        future.get().unwrap()
    }

    fn set_priority(&self, priority: crate::inferlet::core::common::Priority) {
        self.inner.set_priority(priority)
    }

    fn allocate_kv_pages(&self, count: u32) -> Vec<u32> {
        allocate_resources(&self.inner, ResourceType::KvPage as u32, count)
    }

    fn deallocate_kv_pages(&self, ptrs: &[u32]) {
        deallocate_resources(&self.inner, ResourceType::KvPage as u32, ptrs)
    }

    fn export_kv_pages(&self, ptrs: &[u32], name: &str) {
        export_resources(&self.inner, ResourceType::KvPage as u32, ptrs, name)
    }

    fn import_kv_pages(&self, name: &str) -> Vec<u32> {
        import_resources(&self.inner, ResourceType::KvPage as u32, name)
    }

    fn get_all_exported_kv_pages(&self) -> Vec<(String, u32)> {
        get_all_exported_resources(&self.inner, ResourceType::KvPage as u32)
    }

    fn release_exported_kv_pages(&self, name: &str) {
        release_exported_resources(&self.inner, ResourceType::KvPage as u32, name)
    }

    fn allocate_embeds(&self, count: u32) -> Vec<u32> {
        allocate_resources(&self.inner, ResourceType::Embed as u32, count)
    }

    fn deallocate_embeds(&self, ptrs: &[u32]) {
        deallocate_resources(&self.inner, ResourceType::Embed as u32, ptrs)
    }

    fn create_forward_pass(&self) -> ForwardPass {
        ForwardPass {
            inner: Rc::new(create_forward_pass(&self.inner)),
        }
    }
}

impl Clone for Queue {
    fn clone(&self) -> Self {
        Queue {
            inner: Rc::clone(&self.inner),
            service_id: self.service_id,
        }
    }
}

struct ForwardPass {
    inner: Rc<HostForwardPass>,
}

impl ForwardPass {
    async fn execute(&self) -> ForwardPassResult {
        if let Some(future) = self.inner.execute() {
            let pollable = future.pollable();
            AsyncPollable::new(pollable).wait_for().await;

            let mut dists = Vec::new();
            if let Some(distributions) = future.get_distributions() {
                for (ids, probs) in distributions {
                    dists.push(Distribution { ids, probs });
                }
            }
            let distributions = if dists.is_empty() { None } else { Some(dists) };

            ForwardPassResult {
                distributions,
                tokens: future.get_tokens(),
            }
        } else {
            ForwardPassResult {
                distributions: None,
                tokens: None,
            }
        }
    }

    fn input_tokens(&self, tokens: &[u32], positions: &[u32]) {
        input_tokens(&self.inner, tokens, positions);
    }

    fn input_embed_ptrs(&self, embed_ptrs: &[u32], positions: &[u32]) {
        input_embeddings(&self.inner, embed_ptrs, positions);
    }

    fn kv_cache(&self, kv_page_ptrs: &[u32], last_kv_page_len: u32) {
        kv_cache(&self.inner, kv_page_ptrs, last_kv_page_len);
    }

    fn attention_mask(&self, mask: &[Vec<u32>]) {
        attention_mask(&self.inner, mask);
    }

    fn set_adapter(&self, adapter_ptr: u32) {
        set_adapter(&self.inner, adapter_ptr);
    }

    fn set_adapter_seed(&self, seed: i64) {
        set_adapter_seed(&self.inner, seed);
    }

    fn output_distributions(&self, indices: &[u32], temperature: f32, top_k: Option<u32>) {
        output_distributions(&self.inner, indices, temperature, top_k);
    }

    fn output_tokens(&self, indices: &[u32], temperature: f32) {
        output_tokens(&self.inner, indices, temperature);
    }

    fn output_tokens_top_p(&self, indices: &[u32], temperature: f32, top_p: f32) {
        output_tokens_top_p(&self.inner, indices, temperature, top_p);
    }

    fn output_tokens_top_k(&self, indices: &[u32], temperature: f32, top_k: u32) {
        output_tokens_top_k(&self.inner, indices, temperature, top_k);
    }

    fn output_tokens_min_p(&self, indices: &[u32], temperature: f32, min_p: f32) {
        output_tokens_min_p(&self.inner, indices, temperature, min_p);
    }

    fn output_tokens_top_k_top_p(
        &self,
        indices: &[u32],
        temperature: f32,
        top_k: u32,
        top_p: f32,
    ) {
        output_tokens_top_k_top_p(&self.inner, indices, temperature, top_k, top_p);
    }

    fn output_embed_ptrs(&self, embed_ptrs: &[u32], indices: &[u32]) {
        output_embeddings(&self.inner, embed_ptrs, indices);
    }
}

struct QueueImpl {
    inner: RefCell<Queue>,
}

impl GuestQueue for QueueImpl {
    fn from_model_name(model_name: String) -> exports::inferlib2::engine::queues::Queue {
        let host_model = get_model(&model_name).expect("Failed to get model by name");
        let queue = Queue::from_host_model(&host_model);

        exports::inferlib2::engine::queues::Queue::new(QueueImpl {
            inner: RefCell::new(queue),
        })
    }

    fn get_service_id(&self) -> u32 {
        self.inner.borrow().get_service_id()
    }

    fn synchronize(&self) -> bool {
        let inner = self.inner.borrow();
        let inner_clone = inner.clone();
        drop(inner);
        block_on(async move { inner_clone.synchronize().await })
    }

    fn set_priority(&self, priority: Priority) {
        use crate::inferlet::core::common::Priority as HostPriority;
        let host_priority = match priority {
            Priority::Low => HostPriority::Low,
            Priority::Normal => HostPriority::Normal,
            Priority::High => HostPriority::High,
        };
        self.inner.borrow().set_priority(host_priority)
    }

    fn allocate_kv_pages(&self, count: u32) -> Vec<u32> {
        self.inner.borrow().allocate_kv_pages(count)
    }

    fn deallocate_kv_pages(&self, ptrs: Vec<u32>) {
        self.inner.borrow().deallocate_kv_pages(&ptrs)
    }

    fn export_kv_pages(&self, ptrs: Vec<u32>, name: String) {
        self.inner.borrow().export_kv_pages(&ptrs, &name)
    }

    fn import_kv_pages(&self, name: String) -> Vec<u32> {
        self.inner.borrow().import_kv_pages(&name)
    }

    fn get_all_exported_kv_pages(&self) -> Vec<(String, u32)> {
        self.inner.borrow().get_all_exported_kv_pages()
    }

    fn release_exported_kv_pages(&self, name: String) {
        self.inner.borrow().release_exported_kv_pages(&name)
    }

    fn allocate_embeds(&self, count: u32) -> Vec<u32> {
        self.inner.borrow().allocate_embeds(count)
    }

    fn deallocate_embeds(&self, ptrs: Vec<u32>) {
        self.inner.borrow().deallocate_embeds(&ptrs)
    }

    fn create_forward_pass(&self) -> exports::inferlib2::engine::queues::ForwardPass {
        let fp = self.inner.borrow().create_forward_pass();
        exports::inferlib2::engine::queues::ForwardPass::new(ForwardPassImpl {
            inner: RefCell::new(fp),
        })
    }
}

struct ForwardPassImpl {
    inner: RefCell<ForwardPass>,
}

impl GuestForwardPass for ForwardPassImpl {
    fn input_tokens(&self, tokens: Vec<u32>, positions: Vec<u32>) {
        self.inner.borrow().input_tokens(&tokens, &positions)
    }

    fn input_embed_ptrs(&self, embed_ptrs: Vec<u32>, positions: Vec<u32>) {
        self.inner
            .borrow()
            .input_embed_ptrs(&embed_ptrs, &positions)
    }

    fn kv_cache(&self, kv_page_ptrs: Vec<u32>, last_kv_page_len: u32) {
        self.inner
            .borrow()
            .kv_cache(&kv_page_ptrs, last_kv_page_len)
    }

    fn attention_mask(&self, mask: Vec<Vec<u32>>) {
        self.inner.borrow().attention_mask(&mask)
    }

    fn set_adapter(&self, adapter_ptr: u32) {
        self.inner.borrow().set_adapter(adapter_ptr)
    }

    fn set_adapter_seed(&self, seed: i64) {
        self.inner.borrow().set_adapter_seed(seed)
    }

    fn output_distributions(&self, indices: Vec<u32>, temperature: f32, top_k: Option<u32>) {
        self.inner
            .borrow()
            .output_distributions(&indices, temperature, top_k)
    }

    fn output_tokens(&self, indices: Vec<u32>, temperature: f32) {
        self.inner.borrow().output_tokens(&indices, temperature)
    }

    fn output_tokens_top_p(&self, indices: Vec<u32>, temperature: f32, top_p: f32) {
        self.inner
            .borrow()
            .output_tokens_top_p(&indices, temperature, top_p)
    }

    fn output_tokens_top_k(&self, indices: Vec<u32>, temperature: f32, top_k: u32) {
        self.inner
            .borrow()
            .output_tokens_top_k(&indices, temperature, top_k)
    }

    fn output_tokens_min_p(&self, indices: Vec<u32>, temperature: f32, min_p: f32) {
        self.inner
            .borrow()
            .output_tokens_min_p(&indices, temperature, min_p)
    }

    fn output_tokens_top_k_top_p(
        &self,
        indices: Vec<u32>,
        temperature: f32,
        top_k: u32,
        top_p: f32,
    ) {
        self.inner
            .borrow()
            .output_tokens_top_k_top_p(&indices, temperature, top_k, top_p)
    }

    fn output_embed_ptrs(&self, embed_ptrs: Vec<u32>, indices: Vec<u32>) {
        self.inner.borrow().output_embed_ptrs(&embed_ptrs, &indices)
    }

    fn execute(&self) -> ForwardPassResult {
        let inner = self.inner.borrow();
        let inner_rc = Rc::clone(&inner.inner);
        drop(inner);

        block_on(async move {
            let fp = ForwardPass { inner: inner_rc };
            fp.execute().await
        })
    }
}

// ── Runtime implementation ──

use exports::inferlib2::engine::runtime::Guest as RuntimeGuest;

// ── Combined Guest struct ──

struct EngineImpl;

impl ModelsGuest for EngineImpl {
    type Model = ModelImpl;
    type Tokenizer = TokenizerImpl;
}

impl QueuesGuest for EngineImpl {
    type Queue = QueueImpl;
    type ForwardPass = ForwardPassImpl;
}

impl RuntimeGuest for EngineImpl {
    fn get_version() -> String {
        inferlet::core::runtime::get_version()
    }

    fn get_instance_id() -> String {
        inferlet::core::runtime::get_instance_id()
    }

    fn get_arguments() -> Vec<String> {
        inferlet::core::runtime::get_arguments()
    }

    fn set_return(value: String) {
        inferlet::core::runtime::set_return(&value);
    }
}

export!(EngineImpl);
