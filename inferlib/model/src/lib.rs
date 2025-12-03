// Generate WIT bindings for both imports and exports
wit_bindgen::generate!({
    path: "wit",
    world: "model-provider",
    generate_all,
});

use exports::inferlib::model::models::{Guest, GuestModel, GuestTokenizer};

// Import host interfaces from the generated WIT bindings
use crate::inferlet::core::common::Model as HostModel;
use crate::inferlet::core::runtime::{get_all_models, get_model};
use crate::inferlet::core::tokenize::{get_tokenizer, Tokenizer as HostTokenizer};

use std::cell::RefCell;
use std::collections::HashSet;
use std::rc::Rc;

struct ModelsImpl;

impl Guest for ModelsImpl {
    type Model = ModelImpl;
    type Tokenizer = TokenizerImpl;
}

/// Internal Model struct that wraps the host model
pub struct Model {
    inner: Rc<HostModel>,
}

impl Model {
    /// Create a new Model from a host model
    fn from_host(inner: HostModel) -> Self {
        Model {
            inner: Rc::new(inner),
        }
    }

    /// Get a model by name
    pub fn get_by_name(name: &str) -> Option<Self> {
        get_model(name).map(|inner| Model::from_host(inner))
    }

    /// Get the auto-selected model (first available)
    pub fn get_auto() -> Self {
        let models = get_all_models();
        if models.is_empty() {
            panic!("No models available");
        }
        let model_name = &models[0];
        get_model(model_name)
            .map(|inner| Model::from_host(inner))
            .expect("Failed to get first model")
    }

    /// Get all available model names
    pub fn get_all_names() -> Vec<String> {
        get_all_models()
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
        let tokenizer = get_tokenizer(&self.inner);
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

    /// Gets a tokenizer for this model
    pub fn get_tokenizer(&self) -> Tokenizer {
        let host_tokenizer = get_tokenizer(&self.inner);
        Tokenizer {
            inner: Rc::new(host_tokenizer),
        }
    }

    /// Get the inner host model reference (for internal use by other inferlib components)
    pub fn inner(&self) -> &Rc<HostModel> {
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

/// Internal Tokenizer struct that wraps the host tokenizer
pub struct Tokenizer {
    inner: Rc<HostTokenizer>,
}

impl Tokenizer {
    /// Converts a string of text into a sequence of token IDs
    pub fn tokenize(&self, text: &str) -> Vec<u32> {
        self.inner.tokenize(text)
    }

    /// Converts a sequence of token IDs back into a human-readable string
    pub fn detokenize(&self, tokens: &[u32]) -> String {
        self.inner.detokenize(tokens)
    }

    /// Retrieves the entire vocabulary of the tokenizer
    pub fn get_vocabs(&self) -> (Vec<u32>, Vec<Vec<u8>>) {
        self.inner.get_vocabs()
    }
}

// WIT interface wrapper for Model
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

    fn get_tokenizer(&self) -> exports::inferlib::model::models::Tokenizer {
        let tokenizer = self.inner.borrow().get_tokenizer();
        exports::inferlib::model::models::Tokenizer::new(TokenizerImpl {
            inner: RefCell::new(tokenizer),
        })
    }
}

// WIT interface wrapper for Tokenizer
struct TokenizerImpl {
    inner: RefCell<Tokenizer>,
}

impl GuestTokenizer for TokenizerImpl {
    fn tokenize(&self, text: String) -> Vec<u32> {
        self.inner.borrow().tokenize(&text)
    }

    fn detokenize(&self, tokens: Vec<u32>) -> String {
        self.inner.borrow().detokenize(&tokens)
    }

    fn get_vocabs(&self) -> (Vec<u32>, Vec<Vec<u8>>) {
        self.inner.borrow().get_vocabs()
    }
}

export!(ModelsImpl);
