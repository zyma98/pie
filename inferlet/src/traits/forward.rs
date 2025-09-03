use crate::brle::Brle;
use crate::{Queue, Resource, store_get};
use crate::{forward, store_set};
use std::rc::Rc;
use wstd::io::AsyncPollable;

#[derive(Debug, Clone)]
pub struct ForwardPass {
    pub(crate) inner: Rc<forward::ForwardPass>,
}

#[derive(Debug, Clone)]
pub struct ForwardPassResult {
    pub distributions: Option<Vec<(Vec<u32>, Vec<f32>)>>,
    pub tokens: Option<Vec<u32>>,
}

// "Smart" kv page
#[derive(Debug, Clone)]
pub struct KvPage {
    queue: Queue,
    rc: Rc<()>,
    ptr: u32,
}

impl KvPage {
    pub fn new(queue: &Queue, ptr: u32) -> Self {
        KvPage {
            queue: queue.clone(),
            rc: Rc::new(()),
            ptr,
        }
    }

    pub fn ptr(&self) -> u32 {
        self.ptr
    }
}

impl Drop for KvPage {
    fn drop(&mut self) {
        if Rc::strong_count(&self.rc) == 1 {
            self.queue.deallocate_kv_page_ptr(self.ptr);
        }
    }
}

pub fn causal_mask(num_total_tokens: u32, num_input_tokens: u32) -> Vec<Brle> {
    let mut mask = Vec::new();
    let offset = num_total_tokens - num_input_tokens;
    for i in 0..num_input_tokens {
        mask.push(Brle::new((offset + i + 1) as usize));
    }
    mask
}

/// Defines the interface for executing forward passes in a neural network model,
/// including support for PEFT adapters and KV cache manipulation.
pub trait Forward {
    fn new_kv_page(&self) -> KvPage;
    fn new_kv_pages(&self, count: usize) -> Vec<KvPage>;

    fn export_kv_pages(&self, ptrs: &[KvPage], name: &str);
    fn import_kv_pages(&self, name: &str) -> Vec<KvPage>;
    fn allocate_kv_page_ptr(&self) -> u32;
    fn allocate_kv_page_ptrs(&self, count: usize) -> Vec<u32>;
    fn deallocate_kv_page_ptr(&self, ptr: u32);
    fn deallocate_kv_page_ptrs(&self, ptrs: &[u32]);
    fn export_kv_page_ptrs(&self, ptrs: &[u32], name: &str);
    fn import_kv_page_ptrs(&self, name: &str) -> Vec<u32>;

    fn get_all_exported_kv_pages(&self) -> Vec<(String, u32)>;
    fn release_exported_kv_pages(&self, name: &str);

    fn allocate_embed_ptr(&self) -> u32;
    fn allocate_embed_ptrs(&self, count: usize) -> Vec<u32>;
    fn deallocate_embed_ptr(&self, ptr: u32);
    fn deallocate_embed_ptrs(&self, ptrs: &[u32]);
    fn export_embed_ptrs(&self, ptrs: &[u32], name: &str);
    fn import_embed_ptrs(&self, name: &str) -> Vec<u32>;
    fn get_all_exported_embeds(&self) -> Vec<(String, u32)>;
    fn release_exported_embeds(&self, name: &str);

    fn create_forward_pass(&self) -> ForwardPass;
}

impl Forward for Queue {
    fn new_kv_page(&self) -> KvPage {
        let ptr = self.allocate_kv_page_ptr();
        KvPage::new(self, ptr)
    }

    fn new_kv_pages(&self, count: usize) -> Vec<KvPage> {
        self.allocate_kv_page_ptrs(count)
            .into_iter()
            .map(|ptr| KvPage::new(self, ptr))
            .collect()
    }

    fn export_kv_pages(&self, kv_pages: &[KvPage], name: &str) {
        let ptrs = kv_pages.iter().map(|kv| kv.ptr()).collect::<Vec<_>>();
        self.export_resource(Resource::KvPage, &ptrs, name)
    }

    fn import_kv_pages(&self, name: &str) -> Vec<KvPage> {
        let ptrs = self.import_resource(Resource::KvPage, name);
        ptrs.into_iter().map(|ptr| KvPage::new(self, ptr)).collect()
    }

    fn allocate_kv_page_ptr(&self) -> u32 {
        self.allocate_resources(Resource::KvPage, 1)
            .into_iter()
            .next()
            .unwrap()
    }

    fn allocate_kv_page_ptrs(&self, count: usize) -> Vec<u32> {
        self.allocate_resources(Resource::KvPage, count as u32)
    }

    fn deallocate_kv_page_ptr(&self, ptr: u32) {
        self.deallocate_resources(Resource::KvPage, &[ptr])
    }

    fn deallocate_kv_page_ptrs(&self, ptrs: &[u32]) {
        self.deallocate_resources(Resource::KvPage, ptrs)
    }

    fn export_kv_page_ptrs(&self, ptrs: &[u32], name: &str) {
        todo!()
    }

    fn import_kv_page_ptrs(&self, name: &str) -> Vec<u32> {
        todo!()
    }

    fn get_all_exported_kv_pages(&self) -> Vec<(String, u32)> {
        self.get_all_exported_resources(Resource::KvPage)
    }

    fn release_exported_kv_pages(&self, name: &str) {
        self.release_exported_resources(Resource::KvPage, name)
    }

    fn allocate_embed_ptr(&self) -> u32 {
        self.allocate_resources(Resource::Embed, 1)
            .into_iter()
            .next()
            .unwrap()
    }

    fn allocate_embed_ptrs(&self, count: usize) -> Vec<u32> {
        self.allocate_resources(Resource::Embed, count as u32)
    }

    fn deallocate_embed_ptr(&self, ptr: u32) {
        self.deallocate_resources(Resource::Embed, &[ptr])
    }

    fn deallocate_embed_ptrs(&self, ptrs: &[u32]) {
        self.deallocate_resources(Resource::Embed, ptrs)
    }

    fn export_embed_ptrs(&self, ptrs: &[u32], name: &str) {
        self.export_resource(Resource::Embed, ptrs, name)
    }

    fn import_embed_ptrs(&self, name: &str) -> Vec<u32> {
        self.import_resource(Resource::Embed, name)
    }

    fn get_all_exported_embeds(&self) -> Vec<(String, u32)> {
        self.get_all_exported_resources(Resource::Embed)
    }

    fn release_exported_embeds(&self, name: &str) {
        self.release_exported_resources(Resource::Embed, name)
    }

    fn create_forward_pass(&self) -> ForwardPass {
        ForwardPass {
            inner: Rc::new(forward::create_forward_pass(&self.inner)),
        }
    }
}

impl ForwardPass {
    pub async fn execute(&self) -> ForwardPassResult {
        if let Some(future) = self.inner.execute() {
            let pollable = future.pollable();
            AsyncPollable::new(pollable).wait_for().await;
            ForwardPassResult {
                distributions: future.get_distributions(),
                tokens: future.get_tokens(),
            }
        } else {
            ForwardPassResult {
                distributions: None,
                tokens: None,
            }
        }
    }

    pub fn input_embeddings(&self, embed_ptrs: Vec<u32>, positions: Vec<u32>) {
        forward::input_embeddings(&self.inner, &embed_ptrs, &positions);
    }

    pub fn input_tokens(&self, input_tokens: Vec<u32>, positions: Vec<u32>) {
        forward::input_tokens(&self.inner, &input_tokens, &positions);
    }

    pub fn output_embeddings(&self, embed_ptrs: Vec<u32>, indices: Vec<u32>) {
        forward::output_embeddings(&self.inner, &embed_ptrs, &indices);
    }

    pub fn output_distributions(&self, indices: Vec<u32>) {
        forward::output_distributions(&self.inner, &indices);
    }

    pub fn output_tokens(&self, indices: Vec<u32>, samplers: Vec<u32>) {
        forward::output_tokens(&self.inner, &indices, &samplers);
    }

    pub fn attention_mask(&self, mask: Vec<Vec<u32>>) {
        forward::attention_mask(&self.inner, &mask);
    }

    pub fn kv_cache(&self, kv_page_ptrs: Vec<u32>, last_kv_page_len: u32) {
        forward::kv_cache(&self.inner, &kv_page_ptrs, last_kv_page_len);
    }
}
