wit_bindgen::generate!({
    path: "wit",
    world: "importer",
    generate_all,
});

pub use self::inferlib::template::template_rendering::{TemplateRenderer};
pub mod template {
    pub mod template_rendering {
        pub use crate::inferlib::template::template_rendering::*;
    }
}
