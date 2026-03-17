//! Rust bindings for the inferlib-queue WASM component.
//!
//! This crate provides easy-to-use Rust bindings for importing the
//! `inferlib:queue/queues` interface in WASM components.
//!
//! ## Usage
//!
//! ```rust,no_run
//! use inferlib_queue_bindings::{Queue, ForwardPass};
//!
//! // Create a queue from a model name
//! let queue = Queue::from_model_name("my-model");
//!
//! // Allocate KV pages
//! let kv_pages = queue.allocate_kv_pages(4);
//!
//! // Create a forward pass
//! let pass = queue.create_forward_pass();
//! pass.input_tokens(&tokens, &positions);
//! pass.kv_cache(&kv_pages, 0);
//! let result = pass.execute();
//! ```

// Generate WIT bindings
wit_bindgen::generate!({
    path: "wit",
    world: "importer",
    with: {
        "inferlib:queue/queues": generate,
    },
});

// Re-export the main types for convenience
pub use self::inferlib::queue::queues::{
    Distribution, ForwardPass, ForwardPassResult, Priority, Queue, ResourceType,
};

// Re-export the module structure for advanced usage
pub mod queue {
    pub use super::inferlib::queue::*;
}
