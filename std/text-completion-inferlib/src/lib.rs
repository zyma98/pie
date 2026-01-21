use inferlib_context_bindings::Context;
use inferlib_run_bindings::{Args, Result, anyhow};
use inferlib_dummy_bindings::Dummy;

#[inferlib_macros::main]
async fn main(_: Args) -> Result<String> {
    let dummy = Dummy::new();
    let ctx = Context::new(dummy);

    Ok("Hello, world!".to_string())
}
