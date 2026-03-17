use crate::forward::ForwardPass;
use crate::queues::Queue;

use inferlib_engine_bindings::inferlet::adapter::common::set_adapter;
use inferlib_engine_bindings::inferlet::core::common::{
    allocate_resources, deallocate_resources, export_resources, get_all_exported_resources,
    import_resources, release_exported_resources,
};

use crate::exports::inferlib::inference::queues::ResourceType;

impl Queue {
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
        use inferlib_engine_bindings::inferlet::core::common::Blob;
        let blob = Blob::new(data);
        inferlib_engine_bindings::inferlet::adapter::common::upload_adapter(
            &self.inner,
            adapter_ptr,
            name,
            blob,
        );
    }

    pub(crate) fn download_adapter(&self, adapter_ptr: u32, name: &str) {
        inferlib_engine_bindings::inferlet::adapter::common::download_adapter(
            &self.inner,
            adapter_ptr,
            name,
        );
    }
}

impl ForwardPass {
    pub(crate) fn set_adapter(&self, adapter_ptr: u32) {
        set_adapter(&self.inner, adapter_ptr);
    }
}
