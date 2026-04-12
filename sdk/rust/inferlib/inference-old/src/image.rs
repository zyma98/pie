use crate::queues::Queue;

use inferlib_engine_bindings::inferlet::image::image as host_image;

impl Queue {
    /// Embeds an image blob into the provided embedding IDs.
    pub(crate) fn embed_image(&self, embed_ptrs: &[u32], image_data: &[u8], position_offset: u32) {
        host_image::embed_image(&self.inner, embed_ptrs, image_data, position_offset)
    }

    /// Calculates the number of embeddings required for an image of the given dimensions.
    pub(crate) fn calculate_embed_size(&self, image_width: u32, image_height: u32) -> u32 {
        host_image::calculate_embed_size(&self.inner, image_width, image_height)
    }
}
