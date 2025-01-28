use futures::executor::block_on;

//use crate::utils::get_component_linker_store;
//use crate::utils::{bind_interfaces_needed_by_guest_rust_std, ComponentRunStates};
use std::collections::HashMap;
use wasmtime::component::bindgen;
use wasmtime::component::Resource;
use wasmtime::{Engine, Result};

bindgen!({
    path: "../spi/core/wit",
    world: "imports",
    async: true,
    with: {
        "spi:lm/inference/language-model": LanguageModel
    },
    // Interactions with `ResourceTable` can possibly trap so enable the ability
    // to return traps from generated functions.
    trappable_imports: true,
});

pub struct LanguageModel {
    model_id: String,
}