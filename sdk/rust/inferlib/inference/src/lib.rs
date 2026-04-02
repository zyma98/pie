mod adapter;
mod brle;
mod chat;
mod context;
mod formatter;
mod forward;
mod image;
mod kvstore;
mod messaging;
mod models;
mod queues;
mod runtime;
mod zo;

wit_bindgen::generate!({
    path: "wit",
    world: "inference-provider",
    generate_all,
    with: {
        "wasi:io/poll@0.2.0": wasip2::io::poll,
    },
});

use context::{ContextImpl, DecodeStepFutureImpl, FlushFutureImpl, GenerateFutureImpl};
use formatter::ChatFormatterImpl;
use forward::ForwardPassImpl;
use models::{ModelImpl, TokenizerImpl};
use queues::QueueImpl;
use wstd::runtime::{AsyncPollable, block_on};

inferlib_macros::component_bindings!(InferenceComponentImpl);

fn wait_for_pollable(pollable: wasip2::io::poll::Pollable) {
    block_on(async {
        AsyncPollable::new(pollable).wait_for().await;
    });
}
