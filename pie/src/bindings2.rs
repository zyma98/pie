mod allocate;
mod forward;
mod input_image;
mod input_text;
mod model;
mod output_text;
mod runtime;
mod tokenize;

wasmtime::component::bindgen!({
    path: "../inferlet/wit2",
    world: "inferlet",
    async: true,
    with: {
        "wasi:io/poll": wasmtime_wasi::p2::bindings::io::poll,
        "pie:inferlet/runtime/subscription": runtime::Subscription,
        "pie:inferlet/runtime/receive-result": runtime::ReceiveResult,
        "pie:inferlet/model/model": model::Model,
        "pie:inferlet/model/queue": model::Queue,
        "pie:inferlet/model/synchronization-result": model::SynchronizationResult,
        "pie:inferlet/tokenize/tokenizer": tokenize::Tokenizer,
        "pie:inferlet/output-text/distribution-result": output_text::DistributionResult,
    },
    trappable_imports: true,
});
