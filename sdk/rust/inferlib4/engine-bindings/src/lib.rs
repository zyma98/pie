wit_bindgen::generate!({
    path: "wit",
    world: "engine",
    generate_all,
    with: {
        "wasi:io/poll@0.2.4": wasip2::io::poll,
    },
});

pub mod engine {
    pub mod core {
        pub mod common {
            pub use crate::inferlet::core::common::*;
        }
        pub mod runtime {
            pub use crate::inferlet::core::runtime::*;
        }
        pub mod tokenize {
            pub use crate::inferlet::core::tokenize::*;
        }
        pub mod forward {
            pub use crate::inferlet::core::forward::*;
        }
        pub mod kvs {
            pub use crate::inferlet::core::kvs::*;
        }
        pub mod message {
            pub use crate::inferlet::core::message::*;
        }
    }
    pub mod adapter {
        pub mod common {
            pub use crate::inferlet::adapter::common::*;
        }
    }
    pub mod zo {
        pub mod evolve {
            pub use crate::inferlet::zo::evolve::*;
        }
    }
    pub mod image {
        pub mod image {
            pub use crate::inferlet::image::image::*;
        }
    }
}
