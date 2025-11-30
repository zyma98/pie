wit_bindgen::generate!({
    path: "wit",
    world: "brle-bindings",
    generate_all,
});

// Re-export for easier use by downstream crates
pub mod encoding {
    pub use crate::inferlib::brle::encoding::*;
}
pub use encoding::Brle;
