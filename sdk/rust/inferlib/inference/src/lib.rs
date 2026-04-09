mod brle;
mod chat;
mod formatter;
mod inference;
mod kvstore;
mod messaging;
mod models;
mod queues;
mod runtime;

use inferlib_macros::component;
use wstd::runtime::{AsyncPollable, block_on};

component!(InferenceComponentImpl);

fn wait_for_pollable(pollable: wasip2::io::poll::Pollable) {
    block_on(async {
        AsyncPollable::new(pollable).wait_for().await;
    });
}
