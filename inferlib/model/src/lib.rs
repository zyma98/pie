// Generate WIT bindings for exports
wit_bindgen::generate!({
    path: "wit",
    world: "model-provider",
});

use exports::inferlib::model::models::{Guest, GuestModel};

// Import types from the legacy library to access the host API
use inferlet::api;
use std::cell::RefCell;
use std::collections::HashSet;
use std::rc::Rc;

struct ModelsImpl;

impl Guest for ModelsImpl {
    type Model = ModelImpl;
}

/// Internal Model struct that re-implements the legacy Model
pub struct Model {
    inner: Rc<api::Model>,
}

impl Model {
    /// Create a new Model from a host API model
    fn from_api(inner: api::Model) -> Self {
        Model {
            inner: Rc::new(inner),
        }
    }

    /// Get a model by name
    pub fn get_by_name(name: &str) -> Option<Self> {
        api::runtime::get_model(name).map(|inner| Model::from_api(inner))
    }

    /// Get the auto-selected model (first available)
    pub fn get_auto() -> Self {
        let models = api::runtime::get_all_models();
        if models.is_empty() {
            panic!("No models available");
        }
        let model_name = &models[0];
        api::runtime::get_model(model_name)
            .map(|inner| Model::from_api(inner))
            .expect("Failed to get first model")
    }

    /// Get all available model names
    pub fn get_all_names() -> Vec<String> {
        api::runtime::get_all_models()
    }

    /// Returns the model's name (e.g. "llama-3.1-8b-instruct")
    pub fn get_name(&self) -> String {
        self.inner.get_name()
    }

    /// Returns the full set of model traits
    pub fn get_traits(&self) -> Vec<String> {
        self.inner.get_traits()
    }

    /// Check if model has all required traits
    pub fn has_traits(&self, required_traits: &[&str]) -> bool {
        let available_traits_vec = self.get_traits();
        let available_traits: HashSet<&str> =
            available_traits_vec.iter().map(String::as_str).collect();

        // Find any required traits that are not in the available set
        let missing: Vec<_> = required_traits
            .iter()
            .filter(|t| !available_traits.contains(*t))
            .cloned()
            .collect();

        missing.is_empty()
    }

    /// Returns a human-readable description of the model
    pub fn get_description(&self) -> String {
        self.inner.get_description()
    }

    /// Returns the prompt formatting template
    pub fn get_prompt_template(&self) -> String {
        self.inner.get_prompt_template()
    }

    /// Returns EOS token sequences
    pub fn eos_tokens(&self) -> Vec<Vec<u32>> {
        let tokenizer = api::tokenize::get_tokenizer(&self.inner);
        self.inner
            .get_stop_tokens()
            .into_iter()
            .map(|t| tokenizer.tokenize(&t))
            .collect()
    }

    /// Gets the service ID for the model
    pub fn get_service_id(&self) -> u32 {
        self.inner.get_service_id()
    }

    /// Gets the KV page size for the model
    pub fn get_kv_page_size(&self) -> u32 {
        self.inner.get_kv_page_size()
    }

    /// Get the inner API model reference (for internal use by other inferlib components)
    pub fn inner(&self) -> &Rc<api::Model> {
        &self.inner
    }
}

impl Clone for Model {
    fn clone(&self) -> Self {
        Model {
            inner: Rc::clone(&self.inner),
        }
    }
}

// WIT interface wrapper
struct ModelImpl {
    inner: RefCell<Model>,
}

impl GuestModel for ModelImpl {
    fn get_by_name(name: String) -> Option<exports::inferlib::model::models::Model> {
        Model::get_by_name(&name).map(|model| {
            exports::inferlib::model::models::Model::new(ModelImpl {
                inner: RefCell::new(model),
            })
        })
    }

    fn get_auto() -> exports::inferlib::model::models::Model {
        exports::inferlib::model::models::Model::new(ModelImpl {
            inner: RefCell::new(Model::get_auto()),
        })
    }

    fn get_all_names() -> Vec<String> {
        Model::get_all_names()
    }

    fn get_name(&self) -> String {
        self.inner.borrow().get_name()
    }

    fn get_traits(&self) -> Vec<String> {
        self.inner.borrow().get_traits()
    }

    fn has_traits(&self, required_traits: Vec<String>) -> bool {
        let traits_refs: Vec<&str> = required_traits.iter().map(|s| s.as_str()).collect();
        self.inner.borrow().has_traits(&traits_refs)
    }

    fn get_description(&self) -> String {
        self.inner.borrow().get_description()
    }

    fn get_prompt_template(&self) -> String {
        self.inner.borrow().get_prompt_template()
    }

    fn eos_tokens(&self) -> Vec<Vec<u32>> {
        self.inner.borrow().eos_tokens()
    }

    fn get_service_id(&self) -> u32 {
        self.inner.borrow().get_service_id()
    }

    fn get_kv_page_size(&self) -> u32 {
        self.inner.borrow().get_kv_page_size()
    }
}

export!(ModelsImpl);
