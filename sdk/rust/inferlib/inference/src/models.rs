use inferlib_engine_bindings::inferlet::core::common::Model as HostModel;
use inferlib_engine_bindings::inferlet::core::runtime::{get_all_models, get_model};
use inferlib_engine_bindings::inferlet::core::tokenize::{
    Tokenizer as HostTokenizer, get_tokenizer,
};

use std::collections::HashSet;
use std::rc::Rc;

/// Represents a specific model instance, providing access to its metadata and functionality.
#[derive(Clone)]
pub(crate) struct Model {
    pub(crate) inner: Rc<HostModel>,
}

impl Model {
    pub(crate) fn from_host(inner: HostModel) -> Self {
        Model {
            inner: Rc::new(inner),
        }
    }
}

#[derive(Clone)]
pub(crate) struct Tokenizer {
    inner: Rc<HostTokenizer>,
}

impl Model {
    pub(crate) fn get_by_name(name: String) -> Option<Model> {
        get_model(&name).map(Model::from_host)
    }

    pub(crate) fn get_auto() -> Model {
        let models = get_all_models();
        assert!(!models.is_empty(), "No models available");
        get_model(&models[0])
            .map(Model::from_host)
            .expect("Failed to get first model")
    }

    pub(crate) fn get_all_names() -> Vec<String> {
        get_all_models()
    }

    pub(crate) fn get_name(&self) -> String {
        self.inner.get_name()
    }

    pub(crate) fn get_traits(&self) -> Vec<String> {
        self.inner.get_traits()
    }

    pub(crate) fn has_traits(&self, required_traits: Vec<String>) -> bool {
        let traits_refs: Vec<&str> = required_traits.iter().map(|s| s.as_str()).collect();
        let available_traits_vec = self.inner.get_traits();
        let available_traits: HashSet<&str> =
            available_traits_vec.iter().map(String::as_str).collect();
        traits_refs.iter().all(|t| available_traits.contains(t))
    }

    pub(crate) fn get_description(&self) -> String {
        self.inner.get_description()
    }

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

    pub(crate) fn get_service_id(&self) -> u32 {
        self.inner.get_service_id()
    }

    pub(crate) fn get_kv_page_size(&self) -> u32 {
        self.inner.get_kv_page_size()
    }

    pub(crate) fn get_tokenizer(&self) -> Tokenizer {
        Tokenizer {
            inner: Rc::new(get_tokenizer(&self.inner)),
        }
    }
}

impl Tokenizer {
    pub(crate) fn tokenize(&self, text: String) -> Vec<u32> {
        self.inner.tokenize(&text)
    }

    pub(crate) fn detokenize(&self, tokens: Vec<u32>) -> String {
        self.inner.detokenize(&tokens)
    }

    pub(crate) fn get_vocabs(&self) -> (Vec<u32>, Vec<Vec<u8>>) {
        self.inner.get_vocabs()
    }

    pub(crate) fn get_special_tokens(&self) -> (Vec<u32>, Vec<Vec<u8>>) {
        self.inner.get_special_tokens()
    }

    pub(crate) fn get_split_regex(&self) -> String {
        self.inner.get_split_regex()
    }
}
