wit_bindgen::generate!({
    path: "wit",
    world: "importer",
    with: {
        "inferlib2:engine/models": inferlib2_engine_bindings::engine::models,
        "inferlib2:context/inference": generate,
        "inferlib2:context/formatter": generate,
    },
});

// Re-export context/inference types
pub use self::inferlib2::context::inference::{Context, SamplerConfig, StopConfig};

// Re-export context/formatter types
pub use self::inferlib2::context::formatter::{ChatFormatter, ToolCall};

// Re-export Model and Tokenizer from engine-bindings for convenience
pub use inferlib2_engine_bindings::{Model, Tokenizer};

// Re-export module structure for advanced usage
pub mod context {
    pub mod inference {
        pub use crate::inferlib2::context::inference::*;
    }
    pub mod formatter {
        pub use crate::inferlib2::context::formatter::*;
    }
}

pub mod engine {
    pub use inferlib2_engine_bindings::engine::*;
}
