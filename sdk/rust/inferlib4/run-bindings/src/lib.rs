//! Re-exports common types needed by inferlib4 applications and the `#[main]` macro.
//!
//! - `Args` - CLI argument parsing (from pico-args)
//! - `Result`, `anyhow`, etc. - Error handling (from anyhow)
//! - `block_on` - Async runtime (from wstd)
//! - `wit_bindgen` - Used by `inferlib4-macros` to generate export bindings

pub use wit_bindgen;

pub use anyhow::{Context as AnyhowContext, Error, Result, anyhow, bail, ensure, format_err};
pub use pico_args::Arguments as Args;
pub use wstd::runtime::block_on;
