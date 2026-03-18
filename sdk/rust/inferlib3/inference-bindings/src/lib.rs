wit_bindgen::generate!({
    path: "wit",
    world: "importer",
    generate_all,
});

// Re-export model types
pub use self::inferlib3::inference::models::{Model, Tokenizer};

// Re-export queue types
pub use self::inferlib3::inference::queues::{
    Distribution, ForwardPass, ForwardPassResult, Priority, Queue, ResourceType,
};

// Re-export runtime functions
pub use self::inferlib3::inference::runtime::{
    debug_query, get_all_models_with_traits, get_arguments, get_instance_id, get_version,
    set_return,
};

// Re-export inference types
pub use self::inferlib3::inference::inference::{Context, SamplerConfig, StopConfig};

// Re-export formatter types
pub use self::inferlib3::inference::formatter::{ChatFormatter, ToolCall};

// Re-export messaging functions
pub use self::inferlib3::inference::messaging::{
    broadcast, receive, receive_blob, send, send_blob, subscribe,
};

// Re-export kvstore functions
pub use self::inferlib3::inference::kvstore::{
    store_delete, store_exists, store_get, store_list_keys, store_set,
};

// Re-export module structure for advanced usage
pub mod inference {
    pub mod models {
        pub use crate::inferlib3::inference::models::*;
    }
    pub mod queues {
        pub use crate::inferlib3::inference::queues::*;
    }
    pub mod runtime {
        pub use crate::inferlib3::inference::runtime::*;
    }
    pub mod inference {
        pub use crate::inferlib3::inference::inference::*;
    }
    pub mod formatter {
        pub use crate::inferlib3::inference::formatter::*;
    }
    pub mod messaging {
        pub use crate::inferlib3::inference::messaging::*;
    }
    pub mod kvstore {
        pub use crate::inferlib3::inference::kvstore::*;
    }
}
