wit_bindgen::generate!({
    path: "wit",
    world: "testlib-provider",
    generate_all,
});

use exports::testlib::testlib::models::{Guest as ModelsGuest, GuestModel};
use exports::testlib::testlib::inference::{Guest as InferenceGuest, GuestContext};

struct ModelImpl {
    name: String,
}

impl GuestModel for ModelImpl {
    fn get_auto() -> exports::testlib::testlib::models::Model {
        exports::testlib::testlib::models::Model::new(ModelImpl {
            name: "dummy-model".to_string(),
        })
    }

    fn get_name(&self) -> String {
        self.name.clone()
    }
}

struct ContextImpl {
    value: String,
}

impl GuestContext for ContextImpl {
    fn new(wit_model: exports::testlib::testlib::models::ModelBorrow<'_>) -> Self {
        let model_impl: &ModelImpl = wit_model.get();
        let name = model_impl.name.clone();
        ContextImpl {
            value: format!("context-for-{}", name),
        }
    }

    fn get_value(&self) -> String {
        self.value.clone()
    }
}

struct TestlibImpl;

impl ModelsGuest for TestlibImpl {
    type Model = ModelImpl;
}

impl InferenceGuest for TestlibImpl {
    type Context = ContextImpl;
}

export!(TestlibImpl);
