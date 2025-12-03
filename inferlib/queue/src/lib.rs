// Generate WIT bindings for exports
wit_bindgen::generate!({
    path: "wit",
    world: "queue-provider",
});

use exports::inferlib::queue::queues::{
    Distribution, ForwardPassResult, Guest, GuestForwardPass, GuestQueue, Priority, ResourceType,
};

// Import types from the legacy library to access the host API
use inferlet::api;
use inferlet::wstd::runtime::AsyncPollable;
use std::cell::RefCell;
use std::rc::Rc;

struct QueuesImpl;

impl Guest for QueuesImpl {
    type Queue = QueueImpl;
    type ForwardPass = ForwardPassImpl;
}

/// Internal Queue struct that wraps the host API Queue
struct Queue {
    inner: Rc<api::Queue>,
    service_id: u32,
}

impl Queue {
    /// Create a new Queue from a host API model
    fn from_api_model(model: &api::Model) -> Self {
        let queue = model.create_queue();
        let service_id = model.get_service_id();
        Queue {
            inner: Rc::new(queue),
            service_id,
        }
    }

    /// Gets the service ID for the queue
    pub fn get_service_id(&self) -> u32 {
        self.service_id
    }

    /// Synchronize the queue
    pub async fn synchronize(&self) -> bool {
        let future = self.inner.synchronize();
        let pollable = future.pollable();
        AsyncPollable::new(pollable).wait_for().await;
        future.get().unwrap()
    }

    /// Set the queue's priority
    pub fn set_priority(&self, priority: api::Priority) {
        self.inner.set_priority(priority)
    }

    /// Allocate KV page pointers
    pub fn allocate_kv_pages(&self, count: u32) -> Vec<u32> {
        api::allocate_resources(&self.inner, ResourceType::KvPage as u32, count)
    }

    /// Deallocate KV page pointers
    pub fn deallocate_kv_pages(&self, ptrs: &[u32]) {
        api::deallocate_resources(&self.inner, ResourceType::KvPage as u32, ptrs)
    }

    /// Export KV pages with a name
    pub fn export_kv_pages(&self, ptrs: &[u32], name: &str) {
        api::export_resources(&self.inner, ResourceType::KvPage as u32, ptrs, name)
    }

    /// Import KV pages by name
    pub fn import_kv_pages(&self, name: &str) -> Vec<u32> {
        api::import_resources(&self.inner, ResourceType::KvPage as u32, name)
    }

    /// Get all exported KV pages
    pub fn get_all_exported_kv_pages(&self) -> Vec<(String, u32)> {
        api::get_all_exported_resources(&self.inner, ResourceType::KvPage as u32)
    }

    /// Release exported KV pages
    pub fn release_exported_kv_pages(&self, name: &str) {
        api::release_exported_resources(&self.inner, ResourceType::KvPage as u32, name)
    }

    /// Allocate embedding pointers
    pub fn allocate_embeds(&self, count: u32) -> Vec<u32> {
        api::allocate_resources(&self.inner, ResourceType::Embed as u32, count)
    }

    /// Deallocate embedding pointers
    pub fn deallocate_embeds(&self, ptrs: &[u32]) {
        api::deallocate_resources(&self.inner, ResourceType::Embed as u32, ptrs)
    }

    /// Create a forward pass
    pub fn create_forward_pass(&self) -> ForwardPass {
        ForwardPass {
            inner: Rc::new(api::forward::create_forward_pass(&self.inner)),
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

/// Internal ForwardPass struct
struct ForwardPass {
    inner: Rc<api::forward::ForwardPass>,
}

impl ForwardPass {
    /// Execute the forward pass
    pub async fn execute(&self) -> ForwardPassResult {
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

    /// Set input tokens
    pub fn input_tokens(&self, tokens: &[u32], positions: &[u32]) {
        api::forward::input_tokens(&self.inner, tokens, positions);
    }

    /// Set input embedding pointers
    pub fn input_embed_ptrs(&self, embed_ptrs: &[u32], positions: &[u32]) {
        api::forward::input_embeddings(&self.inner, embed_ptrs, positions);
    }

    /// Set KV cache
    pub fn kv_cache(&self, kv_page_ptrs: &[u32], last_kv_page_len: u32) {
        api::forward::kv_cache(&self.inner, kv_page_ptrs, last_kv_page_len);
    }

    /// Set attention mask
    pub fn attention_mask(&self, mask: &[Vec<u32>]) {
        api::forward::attention_mask(&self.inner, mask);
    }

    /// Set adapter
    pub fn set_adapter(&self, adapter_ptr: u32) {
        api::adapter::common::set_adapter(&self.inner, adapter_ptr);
    }

    /// Set adapter seed
    pub fn set_adapter_seed(&self, seed: i64) {
        api::zo::evolve::set_adapter_seed(&self.inner, seed);
    }

    /// Output distributions
    pub fn output_distributions(&self, indices: &[u32], temperature: f32, top_k: Option<u32>) {
        api::forward::output_distributions(&self.inner, indices, temperature, top_k);
    }

    /// Output tokens with multinomial sampling
    pub fn output_tokens(&self, indices: &[u32], temperature: f32) {
        api::forward::output_tokens(&self.inner, indices, temperature);
    }

    /// Output tokens with top-p sampling
    pub fn output_tokens_top_p(&self, indices: &[u32], temperature: f32, top_p: f32) {
        api::forward::output_tokens_top_p(&self.inner, indices, temperature, top_p);
    }

    /// Output tokens with top-k sampling
    pub fn output_tokens_top_k(&self, indices: &[u32], temperature: f32, top_k: u32) {
        api::forward::output_tokens_top_k(&self.inner, indices, temperature, top_k);
    }

    /// Output tokens with min-p sampling
    pub fn output_tokens_min_p(&self, indices: &[u32], temperature: f32, min_p: f32) {
        api::forward::output_tokens_min_p(&self.inner, indices, temperature, min_p);
    }

    /// Output tokens with combined top-k and top-p sampling
    pub fn output_tokens_top_k_top_p(
        &self,
        indices: &[u32],
        temperature: f32,
        top_k: u32,
        top_p: f32,
    ) {
        api::forward::output_tokens_top_k_top_p(&self.inner, indices, temperature, top_k, top_p);
    }

    /// Output embedding pointers
    pub fn output_embed_ptrs(&self, embed_ptrs: &[u32], indices: &[u32]) {
        api::forward::output_embeddings(&self.inner, embed_ptrs, indices);
    }
}

// WIT interface wrapper for Queue
struct QueueImpl {
    inner: RefCell<Queue>,
}

impl GuestQueue for QueueImpl {
    fn from_model_name(model_name: String) -> exports::inferlib::queue::queues::Queue {
        // Look up the host API model by name
        let api_model = api::runtime::get_model(&model_name).expect("Failed to get model by name");
        let queue = Queue::from_api_model(&api_model);

        exports::inferlib::queue::queues::Queue::new(QueueImpl {
            inner: RefCell::new(queue),
        })
    }

    fn get_service_id(&self) -> u32 {
        self.inner.borrow().get_service_id()
    }

    fn synchronize(&self) -> bool {
        let inner = self.inner.borrow();
        let inner_clone = inner.clone();
        drop(inner); // Release borrow before async block
        inferlet::wstd::runtime::block_on(async move { inner_clone.synchronize().await })
    }

    fn set_priority(&self, priority: Priority) {
        let api_priority = match priority {
            Priority::Low => api::Priority::Low,
            Priority::Normal => api::Priority::Normal,
            Priority::High => api::Priority::High,
        };
        self.inner.borrow().set_priority(api_priority)
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

    fn create_forward_pass(&self) -> exports::inferlib::queue::queues::ForwardPass {
        let fp = self.inner.borrow().create_forward_pass();
        exports::inferlib::queue::queues::ForwardPass::new(ForwardPassImpl {
            inner: RefCell::new(fp),
        })
    }
}

// WIT interface wrapper for ForwardPass
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
        // We need to create a new ForwardPass for the async block
        let inner_rc = Rc::clone(&inner.inner);
        drop(inner); // Release borrow before async block

        inferlet::wstd::runtime::block_on(async move {
            let fp = ForwardPass { inner: inner_rc };
            fp.execute().await
        })
    }
}

export!(QueuesImpl);
