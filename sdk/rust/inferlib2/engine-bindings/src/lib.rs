wit_bindgen::generate!({
    path: "wit",
    world: "importer",
    generate_all,
});

// Re-export model types
pub use self::inferlib2::engine::models::{Model, Tokenizer};

// Re-export queue types
pub use self::inferlib2::engine::queues::{
    Distribution, ForwardPass, ForwardPassResult, Priority, Queue, ResourceType,
};

// Re-export runtime functions
pub use self::inferlib2::engine::runtime::{
    get_arguments, get_instance_id, get_version, set_return,
};

// Re-export module structure for advanced usage
pub mod engine {
    pub mod models {
        pub use crate::inferlib2::engine::models::*;
    }
    pub mod queues {
        pub use crate::inferlib2::engine::queues::*;
    }
    pub mod runtime {
        pub use crate::inferlib2::engine::runtime::*;
    }
}
