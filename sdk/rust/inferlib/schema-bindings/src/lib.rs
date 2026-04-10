wit_bindgen::generate!({
    path: "wit",
    world: "importer",
    generate_all,
});

pub use self::inferlib::schema::json_schema::{SchemaValidator};
pub mod schema {
    pub mod json_schema {
        pub use crate::inferlib::schema::json_schema::*;
    }
}
