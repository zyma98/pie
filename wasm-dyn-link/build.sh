#!/bin/bash
set -e

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Building WebAssembly Component Model Dynamic Linking Demo   ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo

# Ensure the wasm32-wasip2 target is installed
echo "=== Checking/installing wasm32-wasip2 target ==="
rustup target add wasm32-wasip2

# Build the provider_logging component (no dependencies)
echo
echo "=== Building provider_logging component ==="
cargo build --package provider_logging --target wasm32-wasip2 --release

# Build the provider_calculator component (depends on provider_logging)
echo
echo "=== Building provider_calculator component ==="
cargo build --package provider_calculator --target wasm32-wasip2 --release

# Build the app consumer component (depends on provider_calculator)
echo
echo "=== Building app_consumer component ==="
cargo build --package app_consumer --target wasm32-wasip2 --release

# Build the host
echo
echo "=== Building host ==="
cargo build --package host --release

# List the built artifacts
echo
echo "=== Built artifacts ==="
ls -la target/wasm32-wasip2/release/*.wasm 2>/dev/null || echo "No wasm files found"
ls -la target/release/host 2>/dev/null || echo "Host binary not found"

echo
echo "=== Build complete! ==="
echo
echo "Run the demo with:"
echo "  ./run.sh"
echo "  or"
echo "  cargo run --package host --release -- \\"
echo "    target/wasm32-wasip2/release/provider_logging.wasm \\"
echo "    target/wasm32-wasip2/release/provider_calculator.wasm \\"
echo "    target/wasm32-wasip2/release/app_consumer.wasm"
