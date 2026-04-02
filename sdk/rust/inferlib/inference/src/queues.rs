use crate::exports::inferlib::inference::queues::{GuestQueue, Priority as WitPriority};
use crate::forward::ForwardPassImpl;
use crate::schema::Priority;

use inferlib_engine_bindings::inferlet::core::common::{Model as HostModel, Queue as HostQueue};
use inferlib_engine_bindings::inferlet::core::runtime::get_model;

use std::cell::RefCell;
use std::rc::Rc;
use wstd::runtime::{AsyncPollable, block_on};

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
        priority: inferlib_engine_bindings::inferlet::core::common::Priority,
    ) {
        self.inner.set_priority(priority)
    }

    pub(crate) async fn debug_query(&self, query: &str) -> String {
        let future = self.inner.debug_query(query);
        let pollable = future.pollable();
        AsyncPollable::new(pollable).wait_for().await;
        future.get().unwrap()
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

pub(crate) struct QueueImpl {
    inner: RefCell<Queue>,
}

impl GuestQueue for QueueImpl {
    fn from_model_name(model_name: String) -> crate::exports::inferlib::inference::queues::Queue {
        let host_model = get_model(&model_name).expect("Failed to get model by name");
        let queue = Queue::from_host_model(&host_model);

        crate::exports::inferlib::inference::queues::Queue::new(QueueImpl {
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

    fn set_priority(&self, priority: WitPriority) {
        use inferlib_engine_bindings::inferlet::core::common::Priority as HostPriority;
        let priority: Priority = priority.into();
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

    fn update_adapter(&self, adapter_ptr: u32, scores: Vec<f32>, seeds: Vec<i64>, max_sigma: f32) {
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

    fn create_forward_pass(&self) -> crate::exports::inferlib::inference::queues::ForwardPass {
        let fp = self.inner.borrow().create_forward_pass();
        crate::exports::inferlib::inference::queues::ForwardPass::new(ForwardPassImpl::new(fp))
    }
}
