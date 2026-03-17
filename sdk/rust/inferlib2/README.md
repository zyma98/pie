# inferlib2

A modular WebAssembly library for LLM inference, built on the WASM Component Model. This is a middle-ground between the monolithic `inferlet` library and the fine-grained `inferlib`.

## Architecture

inferlib2 groups functionality into two coarse-grained WASM components plus a run-bindings crate:

| Component | Package | Exports | Merges from inferlib |
|-----------|---------|---------|----------------------|
| **engine** | `inferlib2:engine` | `models`, `queues`, `runtime` | `model` + `queue` + `environment` |
| **context** | `inferlib2:context` | `inference`, `formatter` | `context` + `chat` + `brle` |
| **run-bindings** | `inferlib:run-bindings` | `inferlet:core/run` | unchanged |

### Crate Types

| Type | Crate Type | Purpose |
|------|------------|---------|
| **WASM Libraries** (`engine`, `context`) | `cdylib` | Compile to `.wasm` components that export WIT interfaces |
| **Rust Bindings** (`engine-bindings`, `context-bindings`) | `rlib` | Provide Rust APIs for importing WIT interfaces in application code |
| **Run Bindings** (`run-bindings`) | `rlib` | Provides the `Guest` trait and `export!` macro for the app entry point |

### Component Diagram

```
Application (e.g. text-completion-inferlib2)
    │
    ├── imports inferlib2:context/inference  (Context)
    ├── imports inferlib2:context/formatter  (ChatFormatter)
    ├── imports inferlib2:engine/models      (Model, Tokenizer)
    └── imports inferlib2:engine/runtime     (get_arguments, set_return)

context component
    │
    ├── exports inferlib2:context/inference
    ├── exports inferlib2:context/formatter
    ├── imports inferlib2:engine/models
    └── imports inferlib2:engine/queues
    (brle is internal, not exported)

engine component
    │
    ├── exports inferlib2:engine/models
    ├── exports inferlib2:engine/queues
    ├── exports inferlib2:engine/runtime
    └── imports inferlet:core/*, inferlet:adapter/*, inferlet:zo/*  (host)
```

## Usage

### Cargo.toml

```toml
[dependencies]
inferlib2-context-bindings = { path = "../../rust/inferlib2/context-bindings" }
inferlib2-engine-bindings = { path = "../../rust/inferlib2/engine-bindings" }
inferlib-run-bindings = { path = "../../rust/inferlib/run-bindings" }
inferlib2-macros = { path = "../../rust/inferlib2-macros" }
```

### Pie.toml

```toml
[dependencies]
inferlib2-engine = "0.1.0"
inferlib2-context = "0.1.0"
```

### Application Code

```rust
use inferlib2_context_bindings::{Context, ChatFormatter, Model, SamplerConfig, StopConfig};
use inferlib_run_bindings::{Args, Result};

#[inferlib2_macros::main]
async fn main(mut args: Args) -> Result<String> {
    let prompt: String = args.value_from_str(["-p", "--prompt"])?;
    let model = Model::get_auto();
    let ctx = Context::new(&model);
    ctx.fill_user(&prompt);
    let response = ctx.generate(SamplerConfig::Greedy, &StopConfig {
        max_tokens: 256,
        eos_sequences: model.eos_tokens(),
    });
    Ok(response)
}
```
