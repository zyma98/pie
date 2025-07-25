use wasmtime::component::HasSelf;
use crate::bindings::l4m;
use crate::bindings::messaging;

mod wit {
    use super::*;
    wasmtime::component::bindgen!({
        path: "../inferlet/wit2",
        world: "inferlet",
        async: true,
        with: {
            "wasi:io/poll": wasmtime_wasi::p2::bindings::io::poll,
            "pie:inferlet/runtime/subscription": messaging::Subscription,
            "pie:inferlet/runtime/receive-result": messaging::ReceiveResult,
            "pie:inferlet/model/model": l4m::Model,
            "pie:inferlet/queue/synchronization-result": l4m::SynchronizationResult,
            "pie:inferlet/tokenize/tokenizer": l4m::Tokenizer,
            "pie:inferlet/output-text/distribution-result": l4m::SampleTopKResult,
        },
        trappable_imports: true,
    });
}

// pub fn add_to_linker<T>(linker: &mut wasmtime::component::Linker<T>) -> Result<(), wasmtime::Error>
// where
//     T: wit::pie::nbi::l4m::Host
//     + wit::pie::nbi::l4m_vision::Host
//     + wit::pie::nbi::runtime::Host
//     + wit::pie::nbi::ping::Host
//     + wit::pie::nbi::messaging::Host,
// {
//     wit::pie::nbi::l4m::add_to_linker::<T, HasSelf<T>>(linker, |s| s)?;
//     wit::pie::nbi::l4m_vision::add_to_linker::<T, HasSelf<T>>(linker, |s| s)?;
//     wit::pie::nbi::runtime::add_to_linker::<T, HasSelf<T>>(linker, |s| s)?;
//     wit::pie::nbi::ping::add_to_linker::<T, HasSelf<T>>(linker, |s| s)?;
//     wit::pie::nbi::messaging::add_to_linker::<T, HasSelf<T>>(linker, |s| s)?;
//     Ok(())
// }
