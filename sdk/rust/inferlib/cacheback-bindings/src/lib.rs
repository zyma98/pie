wit_bindgen::generate!({
    path: "wit",
    world: "importer",
    generate_all,
});

pub use self::inferlib::cacheback::cacheback::{CacheTable, DraftResult};
pub mod cacheback {
    pub mod cacheback {
        pub use crate::inferlib::cacheback::cacheback::*;
    }
}
