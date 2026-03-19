# Inferlib - Modular WASM Inference Library

Inferlib is a modular WebAssembly (WASM) library for LLM inference, designed using the WebAssembly Component Model with WIT (WebAssembly Interface Types) interfaces.

This library is a refactored, modular version of the legacy monolithic [`inferlet`](../inferlet) library. Each module is a separate WASM component that exports well-defined WIT interfaces, promoting better separation of concerns, reusability, and composability.

## Architecture

Inferlib consists of two types of crates:

- **WASM Libraries** (`crate-type = ["cdylib"]`): Compile to `.wasm` components that export WIT interfaces
- **Rust Bindings** (`crate-type = ["rlib"]`): Provide ergonomic Rust APIs for importing WIT interfaces in application code

### Dependency Pattern

When a WASM library depends on another inferlib module, it imports the corresponding bindings crate as a Rust dependency rather than regenerating WIT bindings. This approach:

- **Avoids duplicate code generation** - Bindings are generated once in the `*-bindings` crate
- **Ensures type compatibility** - All consumers share the same generated types
- **Simplifies maintenance** - Changes to an interface only require updating one bindings crate

For example, `context` depends on `model`, `queue`, `chat`, and `brle`. Instead of regenerating bindings for these interfaces, it imports:
- `inferlib-model-bindings`
- `inferlib-queue-bindings`
- `inferlib-chat-bindings`
- `inferlib-brle-bindings`

## Dependency Hierarchy

```
                         ┌───────────────────────────────────┐
                         │           APPLICATION             │
                         │  (e.g., text-completion-inferlib) │
                         └───────────────────────────────────┘
                                          │
                                          ▼
                         ┌───────────────────────────────────┐
                         │             context               │
                         │                                   │
                         │  exports: inferlib:context        │
                         │  depends: model, queue, chat, brle│
                         └───────────────────────────────────┘
                                          │
            ┌─────────────┬───────────────┼───────────────┬─────────────┐
            │             │               │               │             │
            ▼             ▼               ▼               ▼             ▼
     ┌───────────┐ ┌───────────┐  ┌────────────┐  ┌───────────┐ ┌───────────┐
     │   model   │ │   queue   │  │environment │  │   chat    │ │   brle    │
     │           │ │           │  │            │  │           │ │           │
     │ exports:  │ │ exports:  │  │ exports:   │  │ exports:  │ │ exports:  │
     │ inferlib: │ │ inferlib: │  │ inferlib:  │  │ inferlib: │ │ inferlib: │
     │ model     │ │ queue     │  │ environment│  │ chat      │ │ brle      │
     │           │ │           │  │            │  │           │ │           │
     │ imports:  │ │ imports:  │  │ imports:   │  │ (none)    │ │ (none)    │
     │ host only │ │ host only │  │ host only  │  └───────────┘ └───────────┘
     └─────┬─────┘ └─────┬─────┘  └─────┬──────┘
           │             │              │
           └─────────────┼──────────────┘
                         ▼
          ┌─────────────────────────────────────────────────┐
          │                  HOST RUNTIME                   │
          │                                                 │
          │  inferlet:core/*    (common, runtime, tokenize, │
          │                      forward, kvs, message)     │
          │  inferlet:adapter/* (adapter control)           │
          │  inferlet:zo/*      (evolve/mutation)           │
          └─────────────────────────────────────────────────┘
```

## Module Overview

### WASM Libraries

| Module | Package | Description | Crate Dependencies |
|--------|---------|-------------|-------------------|
| **brle** | `inferlib:brle` | Binary Run-Length Encoding for attention masks | None (standalone) |
| **chat** | `inferlib:chat` | Chat message formatting with Jinja2 templates | None (standalone) |
| **environment** | `inferlib:environment` | Runtime environment info (version, args) | None (host only) |
| **model** | `inferlib:model` | Model metadata, tokenizer access | None (host only) |
| **queue** | `inferlib:queue` | Forward pass execution, KV cache management | None (host only) |
| **context** | `inferlib:context` | High-level inference context, generation loops | `model-bindings`, `queue-bindings`, `chat-bindings`, `brle-bindings` |

### Rust Bindings

| Binding | WIT Imports | Crate Dependencies | Description |
|---------|-------------|-------------------|-------------|
| **brle-bindings** | `inferlib:brle/encoding` | None | BRLE operations for Rust apps |
| **chat-bindings** | `inferlib:chat/formatter` | None | Chat formatting for Rust apps |
| **environment-bindings** | `inferlib:environment/runtime` | None | Runtime info for Rust apps |
| **model-bindings** | `inferlib:model/models` | None | Model/tokenizer access for Rust apps |
| **queue-bindings** | `inferlib:queue/queues` | None | Queue operations for Rust apps |
| **context-bindings** | `inferlib:context/inference` | `model-bindings` | Complete inference API for Rust apps |
| **run-bindings** | (exports `inferlet:core/run`) | None | Application entry point, CLI args, async runtime |
