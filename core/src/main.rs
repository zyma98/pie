mod spi;

use wasmtime::Result;
use wasmtime::{Config, Engine};

fn main() -> Result<()> {

    let mut config = Config::default();
    config.async_support(true);

    let engine = Engine::new(&config)?;

    println!("Run without errors!");
    Ok(())
}

fn run_async(engine: &Engine) -> Result<()> {
    unimplemented!("Run async code here")
}