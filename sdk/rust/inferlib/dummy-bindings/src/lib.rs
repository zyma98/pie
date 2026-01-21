wit_bindgen::generate!({
    path: "wit",
    world: "dummy-bindings",
    generate_all,
});

// Re-export for easier use by downstream crates
pub mod greeting {
    pub use crate::inferlib::dummy::greeting::*;
}
pub use greeting::Dummy;
