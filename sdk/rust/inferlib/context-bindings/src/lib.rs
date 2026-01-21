//! Rust bindings for the inferlib-context WASM component.
//!
//! This crate provides easy-to-use Rust bindings for importing the
//! `inferlib:context/inference` interface in WASM components.
//!
//! ## Usage
//!
//! ```rust,no_run
//! use inferlib_context_bindings::{Context, Dummy, SamplerConfig, StopConfig};
//!
//! // Create a dummy
//! let dummy = Dummy::new();
//!
//! // Create a context for the dummy
//! let ctx = Context::new(&dummy);
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

// Generate WIT bindings, using dummy-bindings crate for dummy interface
wit_bindgen::generate!({
    path: "wit",
    world: "importer",
    with: {
        // Use dummy-bindings crate instead of regenerating
        "inferlib:dummy/greeting": inferlib_dummy_bindings::greeting,
        "inferlib:context/inference": generate,
    },
});

// Re-export the main types for convenience
pub use self::inferlib::context::inference::{Context, SamplerConfig, StopConfig};

// Re-export Dummy from dummy-bindings crate
pub use inferlib_dummy_bindings::Dummy;

// Re-export the module structure for advanced usage
pub mod context {
    pub use super::inferlib::context::*;
}

// Re-export dummy module from dummy-bindings for compatibility
pub mod dummy {
    pub use inferlib_dummy_bindings::greeting::*;
}
