// Generate WIT bindings for exports
wit_bindgen::generate!({
    path: "wit",
    world: "context-provider",
    with: {
        "inferlib:dummy/greeting": inferlib_dummy_bindings::greeting,
    },
});

use exports::inferlib::context::inference::{Guest, GuestContext, SamplerConfig, StopConfig};

use inferlib_dummy_bindings::Dummy;

struct InferenceImpl;

impl Guest for InferenceImpl {
    type Context = ContextImpl;
}

// WIT interface wrapper
struct ContextImpl {
}

impl GuestContext for ContextImpl {
    fn new(dummy: Dummy) -> Self {
        ContextImpl {}
    }

    fn fill(&self, _text: String) {}

    fn fill_system(&self, _text: String) {}

    fn fill_user(&self, _text: String) {}

    fn fill_assistant(&self, _text: String) {}

    fn generate(&self, _sampler_config: SamplerConfig, _stop_config: StopConfig) -> String {
        "".to_string()
    }

    fn flush(&self) {}

    fn get_text(&self) -> String {
        "".to_string()
    }

    fn get_token_ids(&self) -> Vec<u32> {
        vec![]
    }
}

export!(InferenceImpl);
