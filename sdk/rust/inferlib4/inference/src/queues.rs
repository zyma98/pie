use crate::exports::inferlib4::inference::queues::{
    Distribution, ForwardPassResult, GuestForwardPass, GuestQueue, Priority, ResourceType,
};

use inferlib4_engine_bindings::inferlet::core::common::{
    Model as HostModel, Queue as HostQueue, allocate_resources, deallocate_resources,
    export_resources, get_all_exported_resources, import_resources, release_exported_resources,
};
use inferlib4_engine_bindings::inferlet::core::forward::{
    ForwardPass as HostForwardPass, attention_mask, create_forward_pass, input_embeddings,
    input_tokens, kv_cache, output_distributions, output_embeddings, output_tokens,
    output_tokens_min_p, output_tokens_top_k, output_tokens_top_k_top_p, output_tokens_top_p,
};
use inferlib4_engine_bindings::inferlet::core::runtime::get_model;
use inferlib4_engine_bindings::inferlet::image::image as host_image;

use inferlib4_engine_bindings::inferlet::adapter::common::set_adapter;
use inferlib4_engine_bindings::inferlet::zo::evolve::set_adapter_seed;

use std::cell::RefCell;
use std::rc::Rc;
use wstd::runtime::{AsyncPollable, block_on};

pub(crate) struct Queue {
    inner: Rc<HostQueue>,
    service_id: u32,
}

impl Queue {
    pub(crate) fn from_host_model(model: &HostModel) -> Self {
        let queue = model.create_queue();
        let service_id = model.get_service_id();
        Queue {
            inner: Rc::new(queue),
            service_id,
        }
    }

    /// Gets the service ID for the queue.
    pub(crate) fn get_service_id(&self) -> u32 {
        self.service_id
    }

    /// Begins a synchronization process for the queue.
    pub(crate) async fn synchronize(&self) -> bool {
        let future = self.inner.synchronize();
        let pollable = future.pollable();
        AsyncPollable::new(pollable).wait_for().await;
        future.get().unwrap()
    }

    /// Change the queue's priority.
    pub(crate) fn set_priority(
        &self,
        priority: inferlib4_engine_bindings::inferlet::core::common::Priority,
    ) {
        self.inner.set_priority(priority)
    }

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

    pub(crate) fn create_forward_pass(&self) -> ForwardPass {
        ForwardPass {
            inner: Rc::new(create_forward_pass(&self.inner)),
        }
    }

    pub(crate) async fn debug_query(&self, query: &str) -> String {
        let future = self.inner.debug_query(query);
        let pollable = future.pollable();
        AsyncPollable::new(pollable).wait_for().await;
        future.get().unwrap()
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

    pub(crate) fn allocate_adapter(&self) -> u32 {
        allocate_resources(&self.inner, ResourceType::Adapter as u32, 1)
            .into_iter()
            .next()
            .unwrap()
    }

    pub(crate) fn deallocate_adapter(&self, ptr: u32) {
        deallocate_resources(&self.inner, ResourceType::Adapter as u32, &[ptr])
    }

    pub(crate) fn export_adapter(&self, ptr: u32, name: &str) {
        export_resources(&self.inner, ResourceType::Adapter as u32, &[ptr], name)
    }

    pub(crate) fn import_adapter(&self, name: &str) -> u32 {
        import_resources(&self.inner, ResourceType::Adapter as u32, name)
            .into_iter()
            .next()
            .unwrap()
    }

    pub(crate) fn get_all_exported_adapters(&self) -> Vec<String> {
        get_all_exported_resources(&self.inner, ResourceType::Adapter as u32)
            .into_iter()
            .map(|(name, _)| name)
            .collect()
    }

    pub(crate) fn release_exported_adapter(&self, name: &str) {
        release_exported_resources(&self.inner, ResourceType::Adapter as u32, name)
    }

    pub(crate) fn upload_adapter(&self, adapter_ptr: u32, name: &str, data: &[u8]) {
        use inferlib4_engine_bindings::inferlet::core::common::Blob;
        let blob = Blob::new(data);
        inferlib4_engine_bindings::inferlet::adapter::common::upload_adapter(
            &self.inner,
            adapter_ptr,
            name,
            blob,
        );
    }

    pub(crate) fn download_adapter(&self, adapter_ptr: u32, name: &str) {
        inferlib4_engine_bindings::inferlet::adapter::common::download_adapter(
            &self.inner,
            adapter_ptr,
            name,
        );
    }

    pub(crate) fn initialize_adapter(
        &self,
        adapter_ptr: u32,
        rank: u32,
        alpha: f32,
        population_size: u32,
        mu_fraction: f32,
        initial_sigma: f32,
    ) {
        inferlib4_engine_bindings::inferlet::zo::evolve::initialize_adapter(
            &self.inner,
            adapter_ptr,
            rank,
            alpha,
            population_size,
            mu_fraction,
            initial_sigma,
        )
    }

    pub(crate) fn update_adapter(
        &self,
        adapter_ptr: u32,
        scores: &[f32],
        seeds: &[i64],
        max_sigma: f32,
    ) {
        inferlib4_engine_bindings::inferlet::zo::evolve::update_adapter(
            &self.inner, adapter_ptr, scores, seeds, max_sigma,
        )
    }

    /// Embeds an image blob into the provided embedding IDs.
    pub(crate) fn embed_image(
        &self,
        embed_ptrs: &[u32],
        image_data: &[u8],
        position_offset: u32,
    ) {
        host_image::embed_image(&self.inner, embed_ptrs, image_data, position_offset)
    }

    /// Calculates the number of embeddings required for an image of the given dimensions.
    pub(crate) fn calculate_embed_size(&self, image_width: u32, image_height: u32) -> u32 {
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

pub(crate) struct ForwardPass {
    pub(crate) inner: Rc<HostForwardPass>,
}

impl ForwardPass {
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

    pub(crate) fn set_adapter(&self, adapter_ptr: u32) {
        set_adapter(&self.inner, adapter_ptr);
    }

    pub(crate) fn set_adapter_seed(&self, seed: i64) {
        set_adapter_seed(&self.inner, seed);
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

pub(crate) struct QueueImpl {
    inner: RefCell<Queue>,
}

impl GuestQueue for QueueImpl {
    fn from_model_name(model_name: String) -> crate::exports::inferlib4::inference::queues::Queue {
        let host_model = get_model(&model_name).expect("Failed to get model by name");
        let queue = Queue::from_host_model(&host_model);

        crate::exports::inferlib4::inference::queues::Queue::new(QueueImpl {
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
        use inferlib4_engine_bindings::inferlet::core::common::Priority as HostPriority;
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

    fn create_forward_pass(&self) -> crate::exports::inferlib4::inference::queues::ForwardPass {
        let fp = self.inner.borrow().create_forward_pass();
        crate::exports::inferlib4::inference::queues::ForwardPass::new(ForwardPassImpl {
            inner: RefCell::new(fp),
        })
    }
}

pub(crate) struct ForwardPassImpl {
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
        self.inner
            .borrow()
            .output_embed_ptrs(&embed_ptrs, &indices)
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
