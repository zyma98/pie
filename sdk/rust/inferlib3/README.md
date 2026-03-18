# inferlib3

A monolithic WASM component library for building inferlets. Unlike `inferlib2` which splits
functionality into separate `engine` and `context` components, `inferlib3` merges everything
into a single `inference` component.

## Architecture

```
inferlib3:inference (single WASM component)
├── models      - Model and Tokenizer resources
├── queues      - Queue and ForwardPass resources
├── runtime     - get-arguments, set-return, etc.
├── inference   - Context resource for generation
└── formatter   - ChatFormatter resource
```

## Crates

| Crate | Type | Description |
|-------|------|-------------|
| `inferlib3-inference` | `cdylib` | Monolithic WASM component exporting all 5 interfaces |
| `inferlib3-inference-bindings` | `rlib` | Rust bindings for importing the component |

## Comparison

| | `inferlet` | `inferlib` | `inferlib2` | `inferlib3` |
|---|---|---|---|---|
| Granularity | Monolithic | Fine (6 components) | Medium (2 components) | Monolithic |
| WASM Component | No | Yes | Yes | Yes |
| Composable | No | Yes | Yes | Yes (single) |
| Components | 0 | 6 | 2 | 1 |

## Usage

Applications depend on `inferlib3-inference-bindings` and `inferlib3-macros`:

```rust
use inferlib3_inference_bindings::{Context, Model, SamplerConfig, StopConfig, ChatFormatter};
use inferlib_run_bindings::{Args, Result};

#[inferlib3_macros::main]
async fn main(mut args: Args) -> Result<String> {
    let model = Model::get_auto();
    let ctx = Context::new(&model);
    // ...
}
```

The `Pie.toml` only needs a single dependency:

```toml
[dependencies]
inferlib3-inference = "0.1.0"
```
