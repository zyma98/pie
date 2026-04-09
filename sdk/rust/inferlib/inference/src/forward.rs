use crate::queues::Queue;

use inferlib_engine_bindings::inferlet::adapter::common::set_adapter;
use inferlib_engine_bindings::inferlet::core::forward::{
    ForwardPass as HostForwardPass, ForwardPassResult as HostForwardPassResult, attention_mask,
    input_embeddings, input_tokens, kv_cache, output_distributions, output_embeddings,
    output_tokens, output_tokens_min_p, output_tokens_top_k, output_tokens_top_k_top_p,
    output_tokens_top_p,
};
use inferlib_engine_bindings::inferlet::zo::evolve::set_adapter_seed;

use std::rc::Rc;
use wstd::runtime::{AsyncPollable, block_on};

inferlib_macros::wit_interface!(queues);

#[derive(Clone, Debug)]
#[inferlib_macros::wit_record]
pub(crate) struct Distribution {
    pub(crate) ids: Vec<u32>,
    pub(crate) probs: Vec<f32>,
}

#[derive(Clone, Debug)]
#[inferlib_macros::wit_record]
pub(crate) struct ForwardPassResult {
    pub(crate) distributions: Option<Vec<Distribution>>,
    pub(crate) tokens: Option<Vec<u32>>,
}

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
            self.queue.deallocate_kv_pages(vec![self.ptr]);
        }
    }
}

pub(crate) struct ForwardPass {
    pub(crate) inner: Rc<HostForwardPass>,
}

impl ForwardPass {
    pub(crate) fn new(inner: HostForwardPass) -> Self {
        ForwardPass {
            inner: Rc::new(inner),
        }
    }

    /// Submits the forward pass and returns the host's raw result handle
    /// without blocking. Use this for async execution paths.
    pub(crate) fn submit_host(&self) -> Option<HostForwardPassResult> {
        self.inner.execute()
    }
}

#[inferlib_macros::guest_resource]
impl ForwardPass {
    pub(crate) fn input_tokens(&self, tokens: Vec<u32>, positions: Vec<u32>) {
        input_tokens(&self.inner, &tokens, &positions);
    }

    pub(crate) fn input_embed_ptrs(&self, embed_ptrs: Vec<u32>, positions: Vec<u32>) {
        input_embeddings(&self.inner, &embed_ptrs, &positions);
    }

    pub(crate) fn kv_cache(&self, kv_page_ptrs: Vec<u32>, last_kv_page_len: u32) {
        kv_cache(&self.inner, &kv_page_ptrs, last_kv_page_len);
    }

    pub(crate) fn attention_mask(&self, mask: Vec<Vec<u32>>) {
        attention_mask(&self.inner, &mask);
    }

    pub(crate) fn set_adapter(&self, adapter_ptr: u32) {
        set_adapter(&self.inner, adapter_ptr);
    }

    pub(crate) fn set_adapter_seed(&self, seed: i64) {
        set_adapter_seed(&self.inner, seed);
    }

    pub(crate) fn output_distributions(
        &self,
        indices: Vec<u32>,
        temperature: f32,
        top_k: Option<u32>,
    ) {
        output_distributions(&self.inner, &indices, temperature, top_k);
    }

    pub(crate) fn output_tokens(&self, indices: Vec<u32>, temperature: f32) {
        output_tokens(&self.inner, &indices, temperature);
    }

    pub(crate) fn output_tokens_top_p(&self, indices: Vec<u32>, temperature: f32, top_p: f32) {
        output_tokens_top_p(&self.inner, &indices, temperature, top_p);
    }

    pub(crate) fn output_tokens_top_k(&self, indices: Vec<u32>, temperature: f32, top_k: u32) {
        output_tokens_top_k(&self.inner, &indices, temperature, top_k);
    }

    pub(crate) fn output_tokens_min_p(&self, indices: Vec<u32>, temperature: f32, min_p: f32) {
        output_tokens_min_p(&self.inner, &indices, temperature, min_p);
    }

    pub(crate) fn output_tokens_top_k_top_p(
        &self,
        indices: Vec<u32>,
        temperature: f32,
        top_k: u32,
        top_p: f32,
    ) {
        output_tokens_top_k_top_p(&self.inner, &indices, temperature, top_k, top_p);
    }

    pub(crate) fn output_embed_ptrs(&self, embed_ptrs: Vec<u32>, indices: Vec<u32>) {
        output_embeddings(&self.inner, &embed_ptrs, &indices);
    }

    pub(crate) fn execute(&self) -> ForwardPassResult {
        if let Some(future) = self.inner.execute() {
            let pollable = future.pollable();
            block_on(async move {
                AsyncPollable::new(pollable).wait_for().await;
            });

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
}
