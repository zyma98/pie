wit_bindgen::generate!({
    path: "wit",
    world: "importer",
    generate_all,
});

pub use self::inferlib4::inference::models::{Model, Tokenizer};

pub use self::inferlib4::inference::queues::{
    Distribution, ForwardPass, ForwardPassResult, Priority, Queue, ResourceType,
};

pub use self::inferlib4::inference::runtime::{
    debug_query, get_all_models_with_traits, get_arguments, get_instance_id, get_version,
    set_return,
};

pub use self::inferlib4::inference::inference::{Context, SamplerConfig, StopConfig};

pub use self::inferlib4::inference::formatter::{ChatFormatter, ToolCall};

pub use self::inferlib4::inference::messaging::{
    broadcast, receive, receive_blob, send, send_blob, subscribe,
};

pub use self::inferlib4::inference::kvstore::{
    store_delete, store_exists, store_get, store_list_keys, store_set,
};

pub mod inference {
    pub mod models {
        pub use crate::inferlib4::inference::models::*;
    }
    pub mod queues {
        pub use crate::inferlib4::inference::queues::*;
    }
    pub mod runtime {
        pub use crate::inferlib4::inference::runtime::*;
    }
    pub mod inference {
        pub use crate::inferlib4::inference::inference::*;
    }
    pub mod formatter {
        pub use crate::inferlib4::inference::formatter::*;
    }
    pub mod messaging {
        pub use crate::inferlib4::inference::messaging::*;
    }
    pub mod kvstore {
        pub use crate::inferlib4::inference::kvstore::*;
    }
}
