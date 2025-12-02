//! Rust bindings for the inferlib-context WASM component.
//!
//! This crate provides easy-to-use Rust bindings for importing the
//! `inferlib:context/inference` interface in WASM components.
//!
//! ## Usage
//!
//! ```rust,no_run
//! use inferlib_context_bindings::{Context, Model, SamplerConfig, StopConfig};
//!
//! // Get a model
//! let model = Model::get_auto();
//!
//! // Create a context for the model
//! let ctx = Context::new(&model);
//!
//! // Fill with messages
//! ctx.fill_system("You are a helpful assistant.");
//! ctx.fill_user("Hello!");
//!
//! // Generate response
//! let stop_config = StopConfig {
//!     max_tokens: 256,
//!     eos_sequences: vec![],
//! };
//! let response = ctx.generate(SamplerConfig::TopP((0.6, 0.95)), &stop_config);
//! ```

// Generate WIT bindings
wit_bindgen::generate!({
    path: "wit",
    world: "importer",
    with: {
        "inferlib:model/models": generate,
        "inferlib:context/inference": generate,
    },
});

// Re-export the main types for convenience
pub use self::inferlib::context::inference::{Context, SamplerConfig, StopConfig};
pub use self::inferlib::model::models::{Model, Tokenizer};

// Re-export the module structure for advanced usage
pub mod context {
    pub use super::inferlib::context::*;
}

pub mod model {
    pub use super::inferlib::model::*;
}
