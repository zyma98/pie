use crate::forward::ForwardPass;

use inferlib_engine_bindings::inferlet::core::common::{
    Model as HostModel, Queue as HostQueue, allocate_resources, deallocate_resources,
    export_resources, get_all_exported_resources, import_resources, release_exported_resources,
};
use inferlib_engine_bindings::inferlet::core::forward::create_forward_pass;
use inferlib_engine_bindings::inferlet::core::runtime::get_model;

use std::rc::Rc;
use wstd::runtime::{AsyncPollable, block_on};

inferlib_macros::wit_interface!(queues);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[inferlib_macros::wit_enum]
pub(crate) enum Priority {
    Low,
    Normal,
    High,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[inferlib_macros::wit_enum]
pub(crate) enum ResourceType {
    KvPage,
    Embed,
    Adapter,
}

#[derive(Clone)]
pub(crate) struct Queue {
    pub(crate) inner: Rc<HostQueue>,
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
}

#[inferlib_macros::guest_resource]
impl Queue {
    pub(crate) fn from_model_name(model_name: String) -> Queue {
        let host_model = get_model(&model_name).expect("Failed to get model by name");
        Queue::from_host_model(&host_model)
    }

    pub(crate) fn get_service_id(&self) -> u32 {
        self.service_id
    }

    pub(crate) fn synchronize(&self) -> bool {
        let future = self.inner.synchronize();
        let pollable = future.pollable();
        block_on(async move {
            AsyncPollable::new(pollable).wait_for().await;
        });
        future.get().unwrap()
    }

    pub(crate) fn set_priority(&self, priority: Priority) {
        use inferlib_engine_bindings::inferlet::core::common::Priority as HostPriority;
        let host_priority = match priority {
            Priority::Low => HostPriority::Low,
            Priority::Normal => HostPriority::Normal,
            Priority::High => HostPriority::High,
        };
        self.inner.set_priority(host_priority)
    }

    pub(crate) fn allocate_kv_pages(&self, count: u32) -> Vec<u32> {
        allocate_resources(&self.inner, ResourceType::KvPage as u32, count)
    }

    pub(crate) fn deallocate_kv_pages(&self, ptrs: Vec<u32>) {
        deallocate_resources(&self.inner, ResourceType::KvPage as u32, &ptrs)
    }

    pub(crate) fn export_kv_pages(&self, ptrs: Vec<u32>, name: String) {
        export_resources(&self.inner, ResourceType::KvPage as u32, &ptrs, &name)
    }

    pub(crate) fn import_kv_pages(&self, name: String) -> Vec<u32> {
        import_resources(&self.inner, ResourceType::KvPage as u32, &name)
    }

    pub(crate) fn get_all_exported_kv_pages(&self) -> Vec<(String, u32)> {
        get_all_exported_resources(&self.inner, ResourceType::KvPage as u32)
    }

    pub(crate) fn release_exported_kv_pages(&self, name: String) {
        release_exported_resources(&self.inner, ResourceType::KvPage as u32, &name)
    }

    pub(crate) fn allocate_embeds(&self, count: u32) -> Vec<u32> {
        allocate_resources(&self.inner, ResourceType::Embed as u32, count)
    }

    pub(crate) fn deallocate_embeds(&self, ptrs: Vec<u32>) {
        deallocate_resources(&self.inner, ResourceType::Embed as u32, &ptrs)
    }

    pub(crate) fn debug_query(&self, query: String) -> String {
        let future = self.inner.debug_query(&query);
        let pollable = future.pollable();
        block_on(async move {
            AsyncPollable::new(pollable).wait_for().await;
        });
        future.get().unwrap()
    }

    pub(crate) fn export_embeds(&self, ptrs: Vec<u32>, name: String) {
        export_resources(&self.inner, ResourceType::Embed as u32, &ptrs, &name)
    }

    pub(crate) fn import_embeds(&self, name: String) -> Vec<u32> {
        import_resources(&self.inner, ResourceType::Embed as u32, &name)
    }

    pub(crate) fn get_all_exported_embeds(&self) -> Vec<(String, u32)> {
        get_all_exported_resources(&self.inner, ResourceType::Embed as u32)
    }

    pub(crate) fn release_exported_embeds(&self, name: String) {
        release_exported_resources(&self.inner, ResourceType::Embed as u32, &name)
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

    pub(crate) fn export_adapter(&self, ptr: u32, name: String) {
        export_resources(&self.inner, ResourceType::Adapter as u32, &[ptr], &name)
    }

    pub(crate) fn import_adapter(&self, name: String) -> u32 {
        import_resources(&self.inner, ResourceType::Adapter as u32, &name)
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

    pub(crate) fn release_exported_adapter(&self, name: String) {
        release_exported_resources(&self.inner, ResourceType::Adapter as u32, &name)
    }

    pub(crate) fn upload_adapter(&self, adapter_ptr: u32, name: String, data: Vec<u8>) {
        use inferlib_engine_bindings::inferlet::core::common::Blob;
        let blob = Blob::new(&data);
        inferlib_engine_bindings::inferlet::adapter::common::upload_adapter(
            &self.inner,
            adapter_ptr,
            &name,
            blob,
        );
    }

    pub(crate) fn download_adapter(&self, adapter_ptr: u32, name: String) {
        inferlib_engine_bindings::inferlet::adapter::common::download_adapter(
            &self.inner,
            adapter_ptr,
            &name,
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
        inferlib_engine_bindings::inferlet::zo::evolve::initialize_adapter(
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
        scores: Vec<f32>,
        seeds: Vec<i64>,
        max_sigma: f32,
    ) {
        inferlib_engine_bindings::inferlet::zo::evolve::update_adapter(
            &self.inner,
            adapter_ptr,
            &scores,
            &seeds,
            max_sigma,
        )
    }

    pub(crate) fn embed_image(
        &self,
        embed_ptrs: Vec<u32>,
        image_data: Vec<u8>,
        position_offset: u32,
    ) {
        inferlib_engine_bindings::inferlet::image::image::embed_image(
            &self.inner,
            &embed_ptrs,
            &image_data,
            position_offset,
        )
    }

    pub(crate) fn calculate_embed_size(&self, image_width: u32, image_height: u32) -> u32 {
        inferlib_engine_bindings::inferlet::image::image::calculate_embed_size(
            &self.inner,
            image_width,
            image_height,
        )
    }

    pub(crate) fn create_forward_pass(&self) -> ForwardPass {
        ForwardPass::new(create_forward_pass(&self.inner))
    }
}
