---
name: build-rust-inferlet
description: How to build a Rust inferlet
---
# Build Rust Inferlet
Detailed instructions for the agent on how to build a Rust inferlet.

## When to Use
- Use this skill when the user asks you to build a Rust inferlet.
- This skill is helpful for users who want to build a Rust inferlet.

## Instructions
- Change directory to the Rust inferlet source directory. This may be the parent directory of the Rust inferlet source directory if the source code is managed as a Rust workspace.
- Run `cargo build --target wasm32-wasip2 --release` to build the Rust inferlet.
- The output will be in the `target/wasm32-wasip2/release` directory.
- The output file will be named `<inferlet-name>.wasm`.
- Both Rust application and library inferlets are built in this way.
