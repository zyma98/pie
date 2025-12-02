//! Rust bindings for the inferlib-model WASM component.
//!
//! This crate provides easy-to-use Rust bindings for importing the
//! `inferlib:model/models` interface in WASM components.
//!
//! ## Usage
//!
//! ```rust,no_run
//! use inferlib_model_bindings::Model;
//!
//! // Get the auto-selected model
//! let model = Model::get_auto();
//!
//! // Get model info
//! let name = model.get_name();
//! let template = model.get_prompt_template();
//! let eos_tokens = model.eos_tokens();
//! ```

// Generate WIT bindings
wit_bindgen::generate!({
    path: "wit",
    world: "importer",
    with: {
        "inferlib:model/models": generate,
    },
});

// Re-export the main types for convenience
pub use self::inferlib::model::models::Model;

// Re-export the module structure for advanced usage
pub mod model {
    pub use super::inferlib::model::*;
}
