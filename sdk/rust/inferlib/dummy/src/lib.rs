// Generate WIT bindings for exports
wit_bindgen::generate!({
    path: "wit",
    world: "dummy-provider",
});

use exports::inferlib::dummy::greeting::{Guest, GuestDummy};

struct GreetingImpl;

impl Guest for GreetingImpl {
    type Dummy = DummyImpl;
}

/// A simple dummy resource for debugging
struct DummyImpl;

impl GuestDummy for DummyImpl {
    fn new() -> Self {
        DummyImpl
    }

    fn hello(&self) -> String {
        "hello world".to_string()
    }
}

export!(GreetingImpl);
