wit_bindgen::generate!({
    path: "wit",
    world: "importer",
    generate_all,
});

pub use self::inferlib::inference::models::{Model, Tokenizer};

pub use self::inferlib::inference::queues::{
    Distribution, ForwardPass, ForwardPassResult, Priority, Queue, ResourceType,
};

pub use self::inferlib::inference::runtime::{
    debug_query, get_all_models_with_traits, get_arguments, get_instance_id, get_version,
    set_return,
};

pub use self::inferlib::inference::inference::{Context, SamplerConfig, StopConfig};

pub use self::inferlib::inference::formatter::{ChatFormatter, ToolCall};

pub use self::inferlib::inference::messaging::{
    broadcast, receive, receive_blob, send, send_blob, subscribe,
};

pub use self::inferlib::inference::kvstore::{
    store_delete, store_exists, store_get, store_list_keys, store_set,
};

pub mod inference {
    pub mod models {
        pub use crate::inferlib::inference::models::*;
    }
    pub mod queues {
        pub use crate::inferlib::inference::queues::*;
    }
    pub mod runtime {
        pub use crate::inferlib::inference::runtime::*;
    }
    pub mod inference {
        pub use crate::inferlib::inference::inference::*;
    }
    pub mod formatter {
        pub use crate::inferlib::inference::formatter::*;
    }
    pub mod messaging {
        pub use crate::inferlib::inference::messaging::*;
    }
    pub mod kvstore {
        pub use crate::inferlib::inference::kvstore::*;
    }
}
