use crate::exports::inferlib::inference::models::{GuestModel, GuestTokenizer};

use inferlib_engine_bindings::inferlet::core::common::Model as HostModel;
use inferlib_engine_bindings::inferlet::core::runtime::{get_all_models, get_model};
use inferlib_engine_bindings::inferlet::core::tokenize::{
    Tokenizer as HostTokenizer, get_tokenizer,
};

use std::cell::RefCell;
use std::collections::HashSet;
use std::rc::Rc;

/// Represents a specific model instance, providing access to its metadata and functionality.
pub(crate) struct Model {
    pub(crate) inner: Rc<HostModel>,
}

impl Model {
    pub(crate) fn from_host(inner: HostModel) -> Self {
        Model {
            inner: Rc::new(inner),
        }
    }

    pub(crate) fn get_by_name(name: &str) -> Option<Self> {
        get_model(name).map(|inner| Model::from_host(inner))
    }

    pub(crate) fn get_auto() -> Self {
        let models = get_all_models();
        if models.is_empty() {
            panic!("No models available");
        }
        let model_name = &models[0];
        get_model(model_name)
            .map(|inner| Model::from_host(inner))
            .expect("Failed to get first model")
    }

    pub(crate) fn get_all_names() -> Vec<String> {
        get_all_models()
    }

    /// Returns the model's name (e.g. "llama-3.1-8b-instruct").
    pub(crate) fn get_name(&self) -> String {
        self.inner.get_name()
    }

    /// Returns the full set of model traits.
    pub(crate) fn get_traits(&self) -> Vec<String> {
        self.inner.get_traits()
    }

    pub(crate) fn has_traits(&self, required_traits: &[&str]) -> bool {
        let available_traits_vec = self.get_traits();
        let available_traits: HashSet<&str> =
            available_traits_vec.iter().map(String::as_str).collect();
        required_traits.iter().all(|t| available_traits.contains(t))
    }

    /// Returns a human-readable description of the model.
    pub(crate) fn get_description(&self) -> String {
        self.inner.get_description()
    }

    /// Returns the prompt formatting template.
    pub(crate) fn get_prompt_template(&self) -> String {
        self.inner.get_prompt_template()
    }

    pub(crate) fn eos_tokens(&self) -> Vec<Vec<u32>> {
        let tokenizer = get_tokenizer(&self.inner);
        self.inner
            .get_stop_tokens()
            .into_iter()
            .map(|t| tokenizer.tokenize(&t))
            .collect()
    }

    /// Gets the service ID for the model.
    pub(crate) fn get_service_id(&self) -> u32 {
        self.inner.get_service_id()
    }

    pub(crate) fn get_kv_page_size(&self) -> u32 {
        self.inner.get_kv_page_size()
    }

    pub(crate) fn get_tokenizer(&self) -> Tokenizer {
        let host_tokenizer = get_tokenizer(&self.inner);
        Tokenizer {
            inner: Rc::new(host_tokenizer),
        }
    }
}

impl Clone for Model {
    fn clone(&self) -> Self {
        Model {
            inner: Rc::clone(&self.inner),
        }
    }
}

pub(crate) struct Tokenizer {
    inner: Rc<HostTokenizer>,
}

impl Tokenizer {
    /// Converts a string of text into a sequence of token IDs.
    pub(crate) fn tokenize(&self, text: &str) -> Vec<u32> {
        self.inner.tokenize(text)
    }

    /// Converts a sequence of token IDs back into a human-readable string.
    pub(crate) fn detokenize(&self, tokens: &[u32]) -> String {
        self.inner.detokenize(tokens)
    }

    /// Retrieves the entire vocabulary of the tokenizer.
    ///
    /// Returns a tuple of (token IDs, token byte sequences).
    pub(crate) fn get_vocabs(&self) -> (Vec<u32>, Vec<Vec<u8>>) {
        self.inner.get_vocabs()
    }

    /// Retrieves the special tokens of the tokenizer.
    ///
    /// Returns a tuple of (special token IDs, special token byte sequences).
    pub(crate) fn get_special_tokens(&self) -> (Vec<u32>, Vec<Vec<u8>>) {
        self.inner.get_special_tokens()
    }

    /// Retrieves the split regular expression of the tokenizer.
    pub(crate) fn get_split_regex(&self) -> String {
        self.inner.get_split_regex()
    }
}

pub(crate) struct ModelImpl {
    pub(crate) inner: RefCell<Model>,
}

impl GuestModel for ModelImpl {
    fn get_by_name(name: String) -> Option<crate::exports::inferlib::inference::models::Model> {
        Model::get_by_name(&name).map(|model| {
            crate::exports::inferlib::inference::models::Model::new(ModelImpl {
                inner: RefCell::new(model),
            })
        })
    }

    fn get_auto() -> crate::exports::inferlib::inference::models::Model {
        crate::exports::inferlib::inference::models::Model::new(ModelImpl {
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

    fn get_tokenizer(&self) -> crate::exports::inferlib::inference::models::Tokenizer {
        let tokenizer = self.inner.borrow().get_tokenizer();
        crate::exports::inferlib::inference::models::Tokenizer::new(TokenizerImpl {
            inner: RefCell::new(tokenizer),
        })
    }
}

pub(crate) struct TokenizerImpl {
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

    fn get_special_tokens(&self) -> (Vec<u32>, Vec<Vec<u8>>) {
        self.inner.borrow().get_special_tokens()
    }

    fn get_split_regex(&self) -> String {
        self.inner.borrow().get_split_regex()
    }
}
