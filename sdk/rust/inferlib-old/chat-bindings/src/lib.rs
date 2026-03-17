//! Rust bindings for the inferlib-chat WASM component.
//!
//! This crate provides easy-to-use Rust bindings for importing the
//! `inferlib:chat/formatter` interface in WASM components.
//!
//! ## Usage
//!
//! ```rust,no_run
//! use inferlib_chat_bindings::ChatFormatter;
//!
//! // Create a formatter instance
//! let formatter = ChatFormatter::new();
//!
//! // Add messages
//! formatter.add_system("You are a helpful assistant.");
//! formatter.add_user("Hello!");
//!
//! // Render the conversation
//! let result = formatter.render(template, true, true);
//! ```

// Generate WIT bindings
wit_bindgen::generate!({
    path: "wit",
    world: "importer",
    with: {
        "inferlib:chat/formatter": generate,
    },
});

// Re-export the main types for convenience
pub use self::inferlib::chat::formatter::{ChatFormatter, ToolCall};

// Re-export the module structure for advanced usage
pub mod chat {
    pub use super::inferlib::chat::*;
}
