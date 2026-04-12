wit_bindgen::generate!({
    path: "wit",
    world: "importer",
    generate_all,
});

pub use self::inferlib::js_engine::js_engine::{execute};
pub mod js_engine {
    pub mod js_engine {
        pub use crate::inferlib::js_engine::js_engine::*;
    }
}
