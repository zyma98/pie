mod brle;
mod chat;

wit_bindgen::generate!({
    path: "wit",
    world: "inference-provider",
    generate_all,
    with: {
        "wasi:io/poll@0.2.4": wasip2::io::poll,
    },
});

use crate::inferlet::image::image as host_image;

// ── Models implementation ──

use exports::inferlib3::inference::models::{
    Guest as ModelsGuest, GuestModel, GuestTokenizer,
};

use crate::inferlet::core::common::Model as HostModel;
use crate::inferlet::core::runtime::{get_all_models, get_model};
use crate::inferlet::core::tokenize::{Tokenizer as HostTokenizer, get_tokenizer};

use std::cell::RefCell;
use std::cmp::Ordering;
use std::collections::HashSet;
use std::mem;
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

    fn get_special_tokens(&self) -> (Vec<u32>, Vec<Vec<u8>>) {
        self.inner.get_special_tokens()
    }

    fn get_split_regex(&self) -> String {
        self.inner.get_split_regex()
    }
}

struct ModelImpl {
    inner: RefCell<Model>,
}

impl GuestModel for ModelImpl {
    fn get_by_name(name: String) -> Option<exports::inferlib3::inference::models::Model> {
        Model::get_by_name(&name).map(|model| {
            exports::inferlib3::inference::models::Model::new(ModelImpl {
                inner: RefCell::new(model),
            })
        })
    }

    fn get_auto() -> exports::inferlib3::inference::models::Model {
        exports::inferlib3::inference::models::Model::new(ModelImpl {
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

    fn get_tokenizer(&self) -> exports::inferlib3::inference::models::Tokenizer {
        let tokenizer = self.inner.borrow().get_tokenizer();
        exports::inferlib3::inference::models::Tokenizer::new(TokenizerImpl {
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

    fn get_special_tokens(&self) -> (Vec<u32>, Vec<Vec<u8>>) {
        self.inner.borrow().get_special_tokens()
    }

    fn get_split_regex(&self) -> String {
        self.inner.borrow().get_split_regex()
    }
}

// ── Queues implementation ──

use exports::inferlib3::inference::queues::{
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

    async fn debug_query(&self, query: &str) -> String {
        let future = self.inner.debug_query(query);
        let pollable = future.pollable();
        AsyncPollable::new(pollable).wait_for().await;
        future.get().unwrap()
    }

    fn export_embeds(&self, ptrs: &[u32], name: &str) {
        export_resources(&self.inner, ResourceType::Embed as u32, ptrs, name)
    }

    fn import_embeds(&self, name: &str) -> Vec<u32> {
        import_resources(&self.inner, ResourceType::Embed as u32, name)
    }

    fn get_all_exported_embeds(&self) -> Vec<(String, u32)> {
        get_all_exported_resources(&self.inner, ResourceType::Embed as u32)
    }

    fn release_exported_embeds(&self, name: &str) {
        release_exported_resources(&self.inner, ResourceType::Embed as u32, name)
    }

    fn allocate_adapter(&self) -> u32 {
        allocate_resources(&self.inner, ResourceType::Adapter as u32, 1)
            .into_iter()
            .next()
            .unwrap()
    }

    fn deallocate_adapter(&self, ptr: u32) {
        deallocate_resources(&self.inner, ResourceType::Adapter as u32, &[ptr])
    }

    fn export_adapter(&self, ptr: u32, name: &str) {
        export_resources(&self.inner, ResourceType::Adapter as u32, &[ptr], name)
    }

    fn import_adapter(&self, name: &str) -> u32 {
        import_resources(&self.inner, ResourceType::Adapter as u32, name)
            .into_iter()
            .next()
            .unwrap()
    }

    fn get_all_exported_adapters(&self) -> Vec<String> {
        get_all_exported_resources(&self.inner, ResourceType::Adapter as u32)
            .into_iter()
            .map(|(name, _)| name)
            .collect()
    }

    fn release_exported_adapter(&self, name: &str) {
        release_exported_resources(&self.inner, ResourceType::Adapter as u32, name)
    }

    fn upload_adapter(&self, adapter_ptr: u32, name: &str, data: &[u8]) {
        use crate::inferlet::core::common::Blob;
        let blob = Blob::new(data);
        crate::inferlet::adapter::common::upload_adapter(&self.inner, adapter_ptr, name, blob);
    }

    fn download_adapter(&self, adapter_ptr: u32, name: &str) {
        crate::inferlet::adapter::common::download_adapter(&self.inner, adapter_ptr, name);
    }

    fn initialize_adapter(
        &self,
        adapter_ptr: u32,
        rank: u32,
        alpha: f32,
        population_size: u32,
        mu_fraction: f32,
        initial_sigma: f32,
    ) {
        crate::inferlet::zo::evolve::initialize_adapter(
            &self.inner,
            adapter_ptr,
            rank,
            alpha,
            population_size,
            mu_fraction,
            initial_sigma,
        )
    }

    fn update_adapter(
        &self,
        adapter_ptr: u32,
        scores: &[f32],
        seeds: &[i64],
        max_sigma: f32,
    ) {
        crate::inferlet::zo::evolve::update_adapter(
            &self.inner, adapter_ptr, scores, seeds, max_sigma,
        )
    }

    fn embed_image(&self, embed_ptrs: &[u32], image_data: &[u8], position_offset: u32) {
        host_image::embed_image(&self.inner, embed_ptrs, image_data, position_offset)
    }

    fn calculate_embed_size(&self, image_width: u32, image_height: u32) -> u32 {
        host_image::calculate_embed_size(&self.inner, image_width, image_height)
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
    fn from_model_name(model_name: String) -> exports::inferlib3::inference::queues::Queue {
        let host_model = get_model(&model_name).expect("Failed to get model by name");
        let queue = Queue::from_host_model(&host_model);

        exports::inferlib3::inference::queues::Queue::new(QueueImpl {
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

    fn debug_query(&self, query: String) -> String {
        let inner = self.inner.borrow();
        let inner_clone = inner.clone();
        drop(inner);
        block_on(async move { inner_clone.debug_query(&query).await })
    }

    fn export_embeds(&self, ptrs: Vec<u32>, name: String) {
        self.inner.borrow().export_embeds(&ptrs, &name)
    }

    fn import_embeds(&self, name: String) -> Vec<u32> {
        self.inner.borrow().import_embeds(&name)
    }

    fn get_all_exported_embeds(&self) -> Vec<(String, u32)> {
        self.inner.borrow().get_all_exported_embeds()
    }

    fn release_exported_embeds(&self, name: String) {
        self.inner.borrow().release_exported_embeds(&name)
    }

    fn allocate_adapter(&self) -> u32 {
        self.inner.borrow().allocate_adapter()
    }

    fn deallocate_adapter(&self, ptr: u32) {
        self.inner.borrow().deallocate_adapter(ptr)
    }

    fn export_adapter(&self, ptr: u32, name: String) {
        self.inner.borrow().export_adapter(ptr, &name)
    }

    fn import_adapter(&self, name: String) -> u32 {
        self.inner.borrow().import_adapter(&name)
    }

    fn get_all_exported_adapters(&self) -> Vec<String> {
        self.inner.borrow().get_all_exported_adapters()
    }

    fn release_exported_adapter(&self, name: String) {
        self.inner.borrow().release_exported_adapter(&name)
    }

    fn upload_adapter(&self, adapter_ptr: u32, name: String, data: Vec<u8>) {
        self.inner
            .borrow()
            .upload_adapter(adapter_ptr, &name, &data)
    }

    fn download_adapter(&self, adapter_ptr: u32, name: String) {
        self.inner.borrow().download_adapter(adapter_ptr, &name)
    }

    fn initialize_adapter(
        &self,
        adapter_ptr: u32,
        rank: u32,
        alpha: f32,
        population_size: u32,
        mu_fraction: f32,
        initial_sigma: f32,
    ) {
        self.inner.borrow().initialize_adapter(
            adapter_ptr,
            rank,
            alpha,
            population_size,
            mu_fraction,
            initial_sigma,
        )
    }

    fn update_adapter(
        &self,
        adapter_ptr: u32,
        scores: Vec<f32>,
        seeds: Vec<i64>,
        max_sigma: f32,
    ) {
        self.inner
            .borrow()
            .update_adapter(adapter_ptr, &scores, &seeds, max_sigma)
    }

    fn embed_image(&self, embed_ptrs: Vec<u32>, image_data: Vec<u8>, position_offset: u32) {
        self.inner
            .borrow()
            .embed_image(&embed_ptrs, &image_data, position_offset)
    }

    fn calculate_embed_size(&self, image_width: u32, image_height: u32) -> u32 {
        self.inner
            .borrow()
            .calculate_embed_size(image_width, image_height)
    }

    fn create_forward_pass(&self) -> exports::inferlib3::inference::queues::ForwardPass {
        let fp = self.inner.borrow().create_forward_pass();
        exports::inferlib3::inference::queues::ForwardPass::new(ForwardPassImpl {
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

use exports::inferlib3::inference::runtime::Guest as RuntimeGuest;

// ── Inference (Context) implementation ──

use exports::inferlib3::inference::inference::{
    Guest as InferenceGuest, GuestContext, SamplerConfig, StopConfig,
};

use brle::Brle;
use chat::ChatFormatter;

struct Context {
    model: Model,
    queue: Queue,
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
            kv_page_ptrs: Vec::new(),
            kv_page_last_len: 0,
            kv_page_size,
            adapter_ptr: None,
            adapter_random_seed: None,
            begin_of_sequence: true,
        }
    }

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
            kv_page_ptrs,
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

    fn get_kv_page_ptrs(&self) -> &[u32] {
        &self.kv_page_ptrs
    }

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
                self.kv_page_ptrs.remove(i);

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

    fn shrink_kv_pages(&mut self, num_tokens: usize) {
        self.adjust_kv_pages(-(num_tokens as isize));
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

        let _ = block_on(async move { p.execute().await });

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

        if let Some(adapter_ptr) = self.adapter_ptr {
            p.set_adapter(adapter_ptr);

            if let Some(adapter_random_seed) = self.adapter_random_seed {
                p.set_adapter_seed(adapter_random_seed);
            }
        }

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

        let res = block_on(async move { p.execute().await });
        let sampled = res.tokens.unwrap().into_iter().next().unwrap();

        self.token_ids.extend(pending_token_ids);
        self.position_ids.extend(position_ids);

        sampled
    }

    fn decode_step_dist(&mut self) -> (Vec<u32>, Vec<f32>) {
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

        p.input_tokens(&pending_token_ids, &position_ids);
        p.kv_cache(&self.kv_page_ptrs, self.kv_page_last_len as u32);
        p.attention_mask(&mask);

        let output_idx = pending_token_ids.len() as u32 - 1;
        p.output_distributions(&[output_idx], 1.0, None);

        let res = block_on(async move { p.execute().await });
        let dist = res.distributions.unwrap().into_iter().next().unwrap();

        self.token_ids.extend(pending_token_ids);
        self.position_ids.extend(position_ids);

        (dist.ids, dist.probs)
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
                self.kv_page_ptrs = beam.kv_page_ptrs.clone();

                return result;
            }

            let mut all_dists = Vec::new();
            for (beam, _, _) in beams.iter_mut() {
                let dist = beam.decode_step_dist();
                all_dists.push(dist);
            }

            let mut next_beams = Vec::new();
            for ((beam, generated, score), (ids, probs)) in
                beams.into_iter().zip(all_dists)
            {
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

    fn fork(&self) -> Self {
        let (
            new_tokens,
            new_pending,
            new_kv_page_ptrs,
            new_kv_page_last_len,
            new_pos_ids,
            new_mask_pending,
        ) = if self.kv_page_last_len == self.kv_page_size {
            (
                self.token_ids.clone(),
                self.token_ids_pending.clone(),
                self.kv_page_ptrs.clone(),
                self.kv_page_last_len,
                self.position_ids.clone(),
                self.token_mask_pending.clone(),
            )
        } else {
            let kept_kv_page_len = self.kv_page_ptrs.len().saturating_sub(1);
            let kept_tokens_len = kept_kv_page_len * self.kv_page_size;

            let forked_token_ids = self.token_ids[..kept_tokens_len].to_vec();
            let forked_kv_page_ptrs = self.kv_page_ptrs[..kept_kv_page_len].to_vec();
            let forked_pos_ids = self.position_ids[..kept_tokens_len].to_vec();

            let forked_pending_token_ids = [
                &self.token_ids[kept_tokens_len..],
                &self.token_ids_pending[..],
            ]
            .concat();

            let forked_last_kv_page_len = if !forked_kv_page_ptrs.is_empty() {
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
                forked_kv_page_ptrs,
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
            kv_page_ptrs: new_kv_page_ptrs,
            kv_page_last_len: new_kv_page_last_len,
            kv_page_size: self.kv_page_size,
            adapter_ptr: self.adapter_ptr,
            adapter_random_seed: self.adapter_random_seed,
            begin_of_sequence: self.begin_of_sequence,
        }
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
    fn new(wit_model: exports::inferlib3::inference::models::ModelBorrow<'_>) -> Self {
        let model_impl: &ModelImpl = wit_model.get();
        let model = model_impl.inner.borrow().clone();
        let inner = Context::new(&model);
        ContextImpl {
            inner: Rc::new(RefCell::new(inner)),
        }
    }

    fn from_imported_state(
        wit_model: exports::inferlib3::inference::models::ModelBorrow<'_>,
        kv_page_ptrs: Vec<u32>,
        prefix_tokens: Vec<u32>,
        kv_page_last_len: u32,
    ) -> exports::inferlib3::inference::inference::Context {
        let model_impl: &ModelImpl = wit_model.get();
        let model = model_impl.inner.borrow().clone();
        let inner =
            Context::from_imported_state(&model, kv_page_ptrs, prefix_tokens, kv_page_last_len as usize);
        exports::inferlib3::inference::inference::Context::new(ContextImpl {
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

    fn fork(&self) -> exports::inferlib3::inference::inference::Context {
        let forked = self.inner.borrow().fork();
        exports::inferlib3::inference::inference::Context::new(ContextImpl {
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
        self.inner.borrow().get_kv_page_ptrs().to_vec()
    }

    fn get_kv_page_last_len(&self) -> u32 {
        self.inner.borrow().get_kv_page_last_len() as u32
    }
}

// ── Chat Formatter WIT export implementation ──

use exports::inferlib3::inference::formatter::{
    Guest as FormatterGuest, GuestChatFormatter, ToolCall as WitToolCall,
};

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

// ── Messaging implementation ──

use exports::inferlib3::inference::messaging::Guest as MessagingGuest;

// ── KVStore implementation ──

use exports::inferlib3::inference::kvstore::Guest as KvstoreGuest;

// ── Combined export ──

struct InferenceComponentImpl;

impl ModelsGuest for InferenceComponentImpl {
    type Model = ModelImpl;
    type Tokenizer = TokenizerImpl;
}

impl QueuesGuest for InferenceComponentImpl {
    type Queue = QueueImpl;
    type ForwardPass = ForwardPassImpl;
}

impl RuntimeGuest for InferenceComponentImpl {
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

    fn get_all_models_with_traits(traits: Vec<String>) -> Vec<String> {
        inferlet::core::runtime::get_all_models_with_traits(&traits)
    }

    fn debug_query(query: String) -> String {
        let future = inferlet::core::runtime::debug_query(&query);
        let pollable = future.pollable();
        block_on(async {
            AsyncPollable::new(pollable).wait_for().await;
        });
        future.get().unwrap()
    }
}

impl InferenceGuest for InferenceComponentImpl {
    type Context = ContextImpl;
}

impl FormatterGuest for InferenceComponentImpl {
    type ChatFormatter = ChatFormatterImpl;
}

impl MessagingGuest for InferenceComponentImpl {
    fn send(message: String) {
        inferlet::core::message::send(&message);
    }

    fn receive() -> String {
        let future = inferlet::core::message::receive();
        let pollable = future.pollable();
        block_on(async {
            AsyncPollable::new(pollable).wait_for().await;
        });
        future.get().unwrap()
    }

    fn send_blob(data: Vec<u8>) {
        use crate::inferlet::core::common::Blob;
        let blob = Blob::new(&data);
        inferlet::core::message::send_blob(blob);
    }

    fn receive_blob() -> Vec<u8> {
        let future = inferlet::core::message::receive_blob();
        let pollable = future.pollable();
        block_on(async {
            AsyncPollable::new(pollable).wait_for().await;
        });
        let blob = future.get().unwrap();
        blob.read(0, blob.size())
    }

    fn broadcast(topic: String, message: String) {
        inferlet::core::message::broadcast(&topic, &message);
    }

    fn subscribe(topic: String) -> String {
        let subscription = inferlet::core::message::subscribe(&topic);
        let pollable = subscription.pollable();
        block_on(async {
            AsyncPollable::new(pollable).wait_for().await;
        });
        subscription.get().unwrap()
    }
}

impl KvstoreGuest for InferenceComponentImpl {
    fn store_get(key: String) -> Option<String> {
        inferlet::core::kvs::store_get(&key)
    }

    fn store_set(key: String, value: String) {
        inferlet::core::kvs::store_set(&key, &value);
    }

    fn store_delete(key: String) {
        inferlet::core::kvs::store_delete(&key);
    }

    fn store_exists(key: String) -> bool {
        inferlet::core::kvs::store_exists(&key)
    }

    fn store_list_keys() -> Vec<String> {
        inferlet::core::kvs::store_list_keys()
    }
}

export!(InferenceComponentImpl);
