use crate::handler::core::Queue;
use crate::handler::{ForwardPassRequest, ForwardPassResponse, Handler};
use crate::instance::InstanceState;
use crate::resource::ResourceId;
use crate::{bindings, model, resource};
use bytes::Bytes;
use tokio::sync::oneshot;
use wasmtime::component::Resource;
use wasmtime_wasi::async_trait;
use wasmtime_wasi::p2::{DynPollable, IoView, Pollable, subscribe};

#[derive(Debug)]
pub struct ForwardPass {
    pub service_id: usize,
    pub stream_id: u32,
    input_tokens: Vec<u32>,
    input_token_positions: Vec<u32>,
    input_embed_ptrs: Vec<u32>,
    input_embed_positions: Vec<u32>,
    pub adapter: Option<u32>,
    pub(crate) adapter_seed: Option<i64>,
    mask: Vec<Vec<u32>>,
    kv_page_ptrs: Vec<u32>,
    kv_page_last_len: u32,
    output_token_indices: Vec<u32>,
    output_token_samplers: Vec<u32>,
    output_dist_indices: Vec<u32>,
    output_embed_ptrs: Vec<u32>,
    output_embed_indices: Vec<u32>,
}

#[derive(Debug)]
pub struct ForwardPassResult {
    pub receiver: oneshot::Receiver<Bytes>,
    pub distributions: Vec<(Vec<u32>, Vec<f32>)>,
    pub tokens: Vec<u32>,
    pub done: bool,
}

#[async_trait]
impl Pollable for ForwardPassResult {
    async fn ready(&mut self) {
        if self.done {
            return;
        }

        let res = (&mut self.receiver).await.unwrap();

        if let Ok(res) = rmp_serde::from_slice::<ForwardPassResponse>(&res) {
            self.distributions = res.dists;
            self.tokens = res.tokens;
        }

        self.done = true;
    }
}

impl bindings::pie::inferlet::forward::Host for InstanceState {
    async fn create_forward_pass(
        &mut self,
        queue: Resource<Queue>,
    ) -> anyhow::Result<Resource<ForwardPass>> {
        let queue = self.table().get(&queue)?.clone();
        let pass = ForwardPass {
            service_id: queue.service_id,
            stream_id: queue.stream_id,
            input_tokens: vec![],
            input_token_positions: vec![],
            input_embed_ptrs: vec![],
            input_embed_positions: vec![],
            adapter: None,
            adapter_seed: None,
            mask: vec![],
            kv_page_ptrs: vec![],
            kv_page_last_len: 0,
            output_token_indices: vec![],
            output_token_samplers: vec![],
            output_dist_indices: vec![],
            output_embed_ptrs: vec![],
            output_embed_indices: vec![],
        };
        Ok(self.table().push(pass)?)
    }

    async fn input_embeddings(
        &mut self,
        pass: Resource<ForwardPass>,
        emb_ptrs: Vec<ResourceId>,
        positions: Vec<u32>,
    ) -> anyhow::Result<()> {
        let svc_id = self.table().get(&pass)?.service_id;
        let phys_emb_ptrs = self
            .translate_resources(svc_id, resource::EMBED_TYPE_ID, &emb_ptrs)
            .ok_or_else(|| {
                anyhow::format_err!(
                    "Failed to translate input embeddings with ptrs: {:?}",
                    emb_ptrs
                )
            })?;

        let pass = self.table().get_mut(&pass)?;
        pass.input_embed_ptrs = phys_emb_ptrs;
        pass.input_embed_positions = positions;
        Ok(())
    }

    async fn input_tokens(
        &mut self,
        pass: Resource<ForwardPass>,
        input_tokens: Vec<u32>,
        positions: Vec<u32>,
    ) -> anyhow::Result<()> {
        let pass = self.table().get_mut(&pass)?;
        pass.input_tokens = input_tokens;
        pass.input_token_positions = positions;
        Ok(())
    }

    async fn output_embeddings(
        &mut self,
        pass: Resource<ForwardPass>,
        emb_ptrs: Vec<ResourceId>,
        indices: Vec<u32>,
    ) -> anyhow::Result<()> {
        let svc_id = self.table().get(&pass)?.service_id;
        let phys_emb_ptrs = self
            .translate_resources(svc_id, resource::EMBED_TYPE_ID, &emb_ptrs)
            .ok_or_else(|| {
                anyhow::format_err!(
                    "Failed to translate input embeddings with ptrs: {:?}",
                    emb_ptrs
                )
            })?;

        let pass = self.table().get_mut(&pass)?;
        pass.output_embed_ptrs = phys_emb_ptrs;
        pass.output_embed_indices = indices;
        Ok(())
    }

    async fn output_distributions(
        &mut self,
        pass: Resource<ForwardPass>,
        indices: Vec<u32>,
    ) -> anyhow::Result<()> {
        let pass = self.table().get_mut(&pass)?;
        pass.output_dist_indices = indices;
        Ok(())
    }

    async fn output_tokens(
        &mut self,
        pass: Resource<ForwardPass>,
        indices: Vec<u32>,
        samplers: Vec<u32>,
    ) -> anyhow::Result<()> {
        let pass = self.table().get_mut(&pass)?;
        pass.output_token_indices = indices;
        pass.output_token_samplers = samplers;
        Ok(())
    }

    async fn attention_mask(
        &mut self,
        pass: Resource<ForwardPass>,
        mask: Vec<Vec<u32>>,
    ) -> anyhow::Result<()> {
        let pass = self.table().get_mut(&pass)?;
        pass.mask = mask;
        Ok(())
    }

    async fn kv_cache(
        &mut self,
        pass: Resource<ForwardPass>,
        kv_page_ptrs: Vec<ResourceId>,
        kv_page_last_len: u32,
    ) -> anyhow::Result<()> {
        let svc_id = self.table().get(&pass)?.service_id;
        let phys_kv_page_ptrs = self
            .translate_resources(svc_id, resource::KV_PAGE_TYPE_ID, &kv_page_ptrs)
            .ok_or_else(|| {
                anyhow::format_err!(
                    "Failed to translate KV cache pages with ptrs: {:?}",
                    kv_page_ptrs
                )
            })?;

        let pass = self.table().get_mut(&pass)?;
        pass.kv_page_ptrs = phys_kv_page_ptrs;
        pass.kv_page_last_len = kv_page_last_len;
        Ok(())
    }
}

impl bindings::pie::inferlet::forward::HostForwardPass for InstanceState {
    async fn execute(
        &mut self,
        this: Resource<ForwardPass>,
    ) -> anyhow::Result<Option<Resource<ForwardPassResult>>> {
        // 1) Check whether we need output (immutable borrow)
        let returns_output = {
            let pass = self.table().get(&this)?;
            !pass.output_dist_indices.is_empty() || !pass.output_token_indices.is_empty()
        };

        // 2) Build the request by MOVING data out of the pass (mutable borrow)
        let (request, service_id, stream_id) = {
            use std::mem::take;

            let pass = self.table().get_mut(&this)?;
            let service_id = pass.service_id;
            let stream_id = pass.stream_id;

            let request = ForwardPassRequest {
                input_tokens: take(&mut pass.input_tokens),
                input_token_positions: take(&mut pass.input_token_positions),
                input_embed_ptrs: take(&mut pass.input_embed_ptrs),
                input_embed_positions: take(&mut pass.input_embed_positions),
                adapter: pass.adapter.unwrap_or(0),
                adapter_seed: pass.adapter_seed.unwrap_or(0),
                mask: take(&mut pass.mask),
                kv_cache_page_ptrs: take(&mut pass.kv_page_ptrs),
                kv_cache_last_page_len: pass.kv_page_last_len,
                output_token_indices: take(&mut pass.output_token_indices),
                output_token_samplers: take(&mut pass.output_token_samplers),
                output_dist_indices: take(&mut pass.output_dist_indices),
                output_embed_ptrs: take(&mut pass.output_embed_ptrs),
                output_embed_indices: take(&mut pass.output_embed_indices),
            };

            (request, service_id, stream_id)
        };

        let data = Bytes::from(rmp_serde::to_vec_named(&request)?);
        let inst_id = self.id();

        if returns_output {
            let (tx, rx) = oneshot::channel();

            model::Command::Submit {
                inst_id,
                cmd_queue_id: stream_id,
                handler: Handler::ForwardPass,
                data,
                response: Some(tx),
            }
            .dispatch(service_id)?;

            let res = ForwardPassResult {
                receiver: rx,
                distributions: vec![],
                tokens: vec![],
                done: false,
            };

            Ok(Some(self.table().push(res)?))
        } else {
            model::Command::Submit {
                inst_id,
                cmd_queue_id: stream_id,
                handler: Handler::ForwardPass,
                data,
                response: None,
            }
            .dispatch(service_id)?;

            Ok(None)
        }
    }
    async fn drop(&mut self, this: Resource<ForwardPass>) -> anyhow::Result<()> {
        self.table().delete(this)?;
        Ok(())
    }
}

impl bindings::pie::inferlet::forward::HostForwardPassResult for InstanceState {
    async fn pollable(
        &mut self,
        this: Resource<ForwardPassResult>,
    ) -> anyhow::Result<Resource<DynPollable>> {
        subscribe(self.table(), this)
    }

    async fn get_distributions(
        &mut self,
        this: Resource<ForwardPassResult>,
    ) -> anyhow::Result<Option<Vec<(Vec<u32>, Vec<f32>)>>> {
        let result = self.table().get_mut(&this)?;

        if result.done {
            Ok(Some(std::mem::take(&mut result.distributions)))
        } else {
            Ok(None)
        }
    }

    async fn get_tokens(
        &mut self,
        this: Resource<ForwardPassResult>,
    ) -> anyhow::Result<Option<Vec<u32>>> {
        let result = self.table().get_mut(&this)?;

        if result.done {
            Ok(Some(std::mem::take(&mut result.tokens)))
        } else {
            Ok(None)
        }
    }

    async fn drop(&mut self, this: Resource<ForwardPassResult>) -> anyhow::Result<()> {
        self.table().delete(this)?;
        Ok(())
    }
}
