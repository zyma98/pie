//! Rust bindings for the inferlib-environment WASM component.
//!
//! This crate provides easy-to-use Rust bindings for importing the
//! `inferlib:environment/runtime` interface in WASM components.
//!
//! ## Usage
//!
//! ```rust,no_run
//! use inferlib_environment_bindings::{get_version, get_instance_id, get_arguments, set_return};
//!
//! // Get runtime version
//! let version = get_version();
//!
//! // Get instance ID
//! let instance_id = get_instance_id();
//!
//! // Get CLI arguments
//! let args = get_arguments();
//!
//! // Set return value
//! set_return("result");
//! ```

// Generate WIT bindings
wit_bindgen::generate!({
    path: "wit",
    world: "importer",
    with: {
        "inferlib:environment/runtime": generate,
    },
});

// Re-export the functions for convenience
pub use self::inferlib::environment::runtime::{
    get_arguments, get_instance_id, get_version, set_return,
};

// Re-export the module structure for advanced usage
pub mod environment {
    pub use super::inferlib::environment::*;
}
