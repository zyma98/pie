//! Rust bindings for the inferlib application entry point.
//!
//! This crate provides the `Guest` trait and `export!` macro for building
//! WASM applications that implement the `inferlet:core/run` interface.
//!
//! It also re-exports common types needed for application development:
//! - `Args` - CLI argument parsing (from pico-args)
//! - `Result`, `anyhow`, etc. - Error handling (from anyhow)
//! - `block_on` - Async runtime (from wstd)
//!
//! ## Usage
//!
//! ```rust,no_run
//! use inferlib_run_bindings::{Args, Result, anyhow, block_on, Guest, export};
//!
//! struct MyApp;
//!
//! impl Guest for MyApp {
//!     fn run() -> Result<(), String> {
//!         // Your application logic here
//!         Ok(())
//!     }
//! }
//!
//! export!(MyApp with_types_in inferlib_run_bindings);
//! ```

// Generate WIT bindings for the app world
wit_bindgen::generate!({
    path: "wit",
    world: "app",
    pub_export_macro: true,
    generate_all,
});

// Re-export the Guest trait for applications to implement
pub use exports::inferlet::core::run::Guest;

// Re-export common types for application development
pub use anyhow::{anyhow, bail, ensure, format_err, Context as AnyhowContext, Error, Result};
pub use pico_args::Arguments as Args;
pub use wstd::runtime::block_on;
