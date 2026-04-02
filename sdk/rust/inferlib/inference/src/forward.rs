use crate::exports::inferlib::inference::queues::{
    Distribution, ForwardPassResult, GuestForwardPass,
};
use crate::queues::Queue;
use crate::schema::ResourceType;

use inferlib_engine_bindings::inferlet::core::common::{
    allocate_resources, deallocate_resources, export_resources, get_all_exported_resources,
    import_resources, release_exported_resources,
};
use inferlib_engine_bindings::inferlet::core::forward::{
    ForwardPass as HostForwardPass, ForwardPassResult as HostForwardPassResult, attention_mask,
    create_forward_pass, input_embeddings, input_tokens, kv_cache, output_distributions,
    output_embeddings, output_tokens, output_tokens_min_p, output_tokens_top_k,
    output_tokens_top_k_top_p, output_tokens_top_p,
};

use std::cell::RefCell;
use std::rc::Rc;
use wstd::runtime::{AsyncPollable, block_on};

/// A reference-counted KV cache page. Cloning increments the reference count;
/// the underlying page is only deallocated when the last reference is dropped.
#[derive(Clone)]
pub(crate) struct KvPage {
    queue: Queue,
    rc: Rc<()>,
    ptr: u32,
}

impl KvPage {
    pub(crate) fn new(queue: &Queue, ptr: u32) -> Self {
        KvPage {
            queue: queue.clone(),
            rc: Rc::new(()),
            ptr,
        }
    }

    pub(crate) fn ptr(&self) -> u32 {
        self.ptr
    }
}

impl Drop for KvPage {
    fn drop(&mut self) {
        if Rc::strong_count(&self.rc) == 1 {
            self.queue.deallocate_kv_pages(&[self.ptr]);
        }
    }
}

impl Queue {
    pub(crate) fn allocate_kv_pages(&self, count: u32) -> Vec<u32> {
        allocate_resources(&self.inner, ResourceType::KvPage as u32, count)
    }

    pub(crate) fn deallocate_kv_pages(&self, ptrs: &[u32]) {
        deallocate_resources(&self.inner, ResourceType::KvPage as u32, ptrs)
    }

    pub(crate) fn export_kv_pages(&self, ptrs: &[u32], name: &str) {
        export_resources(&self.inner, ResourceType::KvPage as u32, ptrs, name)
    }

    pub(crate) fn import_kv_pages(&self, name: &str) -> Vec<u32> {
        import_resources(&self.inner, ResourceType::KvPage as u32, name)
    }

    pub(crate) fn get_all_exported_kv_pages(&self) -> Vec<(String, u32)> {
        get_all_exported_resources(&self.inner, ResourceType::KvPage as u32)
    }

    pub(crate) fn release_exported_kv_pages(&self, name: &str) {
        release_exported_resources(&self.inner, ResourceType::KvPage as u32, name)
    }

    pub(crate) fn allocate_embeds(&self, count: u32) -> Vec<u32> {
        allocate_resources(&self.inner, ResourceType::Embed as u32, count)
    }

    pub(crate) fn deallocate_embeds(&self, ptrs: &[u32]) {
        deallocate_resources(&self.inner, ResourceType::Embed as u32, ptrs)
    }

    pub(crate) fn export_embeds(&self, ptrs: &[u32], name: &str) {
        export_resources(&self.inner, ResourceType::Embed as u32, ptrs, name)
    }

    pub(crate) fn import_embeds(&self, name: &str) -> Vec<u32> {
        import_resources(&self.inner, ResourceType::Embed as u32, name)
    }

    pub(crate) fn get_all_exported_embeds(&self) -> Vec<(String, u32)> {
        get_all_exported_resources(&self.inner, ResourceType::Embed as u32)
    }

    pub(crate) fn release_exported_embeds(&self, name: &str) {
        release_exported_resources(&self.inner, ResourceType::Embed as u32, name)
    }

    pub(crate) fn create_forward_pass(&self) -> ForwardPass {
        ForwardPass {
            inner: Rc::new(create_forward_pass(&self.inner)),
        }
    }
}

pub(crate) struct ForwardPass {
    pub(crate) inner: Rc<HostForwardPass>,
}

impl ForwardPass {
    /// Submits the forward pass and returns the host's raw result handle
    /// without blocking. Use this for async execution paths.
    pub(crate) fn submit(&self) -> Option<HostForwardPassResult> {
        self.inner.execute()
    }

    pub(crate) async fn execute(&self) -> ForwardPassResult {
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

    pub(crate) fn input_tokens(&self, tokens: &[u32], positions: &[u32]) {
        input_tokens(&self.inner, tokens, positions);
    }

    pub(crate) fn input_embed_ptrs(&self, embed_ptrs: &[u32], positions: &[u32]) {
        input_embeddings(&self.inner, embed_ptrs, positions);
    }

    pub(crate) fn kv_cache(&self, kv_page_ptrs: &[u32], last_kv_page_len: u32) {
        kv_cache(&self.inner, kv_page_ptrs, last_kv_page_len);
    }

    pub(crate) fn attention_mask(&self, mask: &[Vec<u32>]) {
        attention_mask(&self.inner, mask);
    }

    pub(crate) fn output_distributions(
        &self,
        indices: &[u32],
        temperature: f32,
        top_k: Option<u32>,
    ) {
        output_distributions(&self.inner, indices, temperature, top_k);
    }

    pub(crate) fn output_tokens(&self, indices: &[u32], temperature: f32) {
        output_tokens(&self.inner, indices, temperature);
    }

    pub(crate) fn output_tokens_top_p(&self, indices: &[u32], temperature: f32, top_p: f32) {
        output_tokens_top_p(&self.inner, indices, temperature, top_p);
    }

    pub(crate) fn output_tokens_top_k(&self, indices: &[u32], temperature: f32, top_k: u32) {
        output_tokens_top_k(&self.inner, indices, temperature, top_k);
    }

    pub(crate) fn output_tokens_min_p(&self, indices: &[u32], temperature: f32, min_p: f32) {
        output_tokens_min_p(&self.inner, indices, temperature, min_p);
    }

    pub(crate) fn output_tokens_top_k_top_p(
        &self,
        indices: &[u32],
        temperature: f32,
        top_k: u32,
        top_p: f32,
    ) {
        output_tokens_top_k_top_p(&self.inner, indices, temperature, top_k, top_p);
    }

    pub(crate) fn output_embed_ptrs(&self, embed_ptrs: &[u32], indices: &[u32]) {
        output_embeddings(&self.inner, embed_ptrs, indices);
    }
}

pub(crate) struct ForwardPassImpl {
    inner: RefCell<ForwardPass>,
}

impl ForwardPassImpl {
    pub(crate) fn new(fp: ForwardPass) -> Self {
        ForwardPassImpl {
            inner: RefCell::new(fp),
        }
    }
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
