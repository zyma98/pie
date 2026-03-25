#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
MICROBENCH_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ENGINE_LOG=$(mktemp)
ENGINE_PID=""
ECHO_CALLEE_BINDINGS="$MICROBENCH_DIR/func-call-latency/inferlib-r2r-echo-callee-bindings"
P2R_SYMLINK="$REPO_ROOT/sdk/rust/inferlib/p2r-echo-callee-bindings"
P2P_SYMLINK="$REPO_ROOT/sdk/rust/inferlib/p2p-echo-callee-bindings"

cleanup() {
    if [ -n "$ENGINE_PID" ] && kill -0 "$ENGINE_PID" 2>/dev/null; then
        echo ""
        echo "=== Stopping engine (PID $ENGINE_PID) ==="
        kill -TERM "$ENGINE_PID"
        wait "$ENGINE_PID" 2>/dev/null || true
        echo "Engine stopped."
    fi
    rm -f "$ENGINE_LOG"
    rm -f "$P2R_SYMLINK"
    rm -f "$P2P_SYMLINK"
}
trap cleanup EXIT

source "$REPO_ROOT/pie/.venv/bin/activate"

echo "=== Building runtime with microbench_call_latency feature ==="
cd "$REPO_ROOT/pie"
if ! maturin develop --release --features microbench_call_latency > /dev/null 2>&1; then
    echo "maturin build failed! Re-running with output:"
    maturin develop --release --features microbench_call_latency
    exit 1
fi

echo "=== Building inferlib dependencies ==="
cd "$REPO_ROOT/sdk/rust/inferlib"
cargo build -rq --target wasm32-wasip2 2>/dev/null

echo "=== Building microbench crates ==="
cd "$MICROBENCH_DIR"
cargo build -rq --target wasm32-wasip2 2>/dev/null

echo "=== Building Python echo callees ==="
PY_CALLEE_OUT="$MICROBENCH_DIR/target/wasm32-wasip2/release"
componentize-py -d "$SCRIPT_DIR/inferlib-p2p-echo-callee/wit" -w echo-provider \
    componentize --no-snapshot --shared-modules auto \
    -e "$SCRIPT_DIR/inferlib-p2p-echo-callee" \
    -o "$PY_CALLEE_OUT/inferlib_p2p_echo_callee.wasm" app
componentize-py -d "$SCRIPT_DIR/inferlib-r2p-echo-callee/wit" -w echo-provider \
    componentize --no-snapshot --shared-modules auto \
    -e "$SCRIPT_DIR/inferlib-r2p-echo-callee" \
    -o "$PY_CALLEE_OUT/inferlib_r2p_echo_callee.wasm" app

echo "=== Building Python echo callers (P2R, P2P) ==="
ln -sfn "$ECHO_CALLEE_BINDINGS" "$P2R_SYMLINK"
ln -sfn "$ECHO_CALLEE_BINDINGS" "$P2P_SYMLINK"
cd "$MICROBENCH_DIR"
bakery build --inferlib func-call-latency/inferlib-p2r-echo-caller \
    -o target/wasm32-wasip2/release/inferlib_p2r_echo_caller.wasm
bakery build --inferlib func-call-latency/inferlib-p2p-echo-caller \
    -o target/wasm32-wasip2/release/inferlib_p2p_echo_caller.wasm
rm -f "$P2R_SYMLINK" "$P2P_SYMLINK"

echo "=== Starting engine (--no-snapshot) ==="
cd "$REPO_ROOT/pie"
pie serve --no-snapshot > "$ENGINE_LOG" 2>&1 &
ENGINE_PID=$!

echo -n "Waiting for engine to start..."
while ! grep -q "Engine running" "$ENGINE_LOG" 2>/dev/null; do
    if ! kill -0 "$ENGINE_PID" 2>/dev/null; then
        echo " FAILED"
        echo "Engine exited unexpectedly. Log output:"
        cat "$ENGINE_LOG"
        exit 1
    fi
    sleep 1
    echo -n "."
done
echo " ready (PID $ENGINE_PID)"

echo "=== Installing inferlib-inference ==="
cd "$REPO_ROOT/sdk/rust/inferlib"
pie-client install \
    --path target/wasm32-wasip2/release/inferlib_inference.wasm \
    --manifest inference/Pie.toml

echo "=== Installing inferlib-r2r-echo-callee ==="
cd "$MICROBENCH_DIR"
pie-client install \
    --path target/wasm32-wasip2/release/inferlib_r2r_echo_callee.wasm \
    --manifest func-call-latency/inferlib-r2r-echo-callee/Pie.toml

echo "=== Installing inferlib-p2r-echo-callee ==="
pie-client install \
    --path target/wasm32-wasip2/release/inferlib_p2r_echo_callee.wasm \
    --manifest func-call-latency/inferlib-p2r-echo-callee/Pie.toml

echo "=== Installing inferlib-p2p-echo-callee ==="
pie-client install \
    --path target/wasm32-wasip2/release/inferlib_p2p_echo_callee.wasm \
    --manifest func-call-latency/inferlib-p2p-echo-callee/Pie.toml

echo "=== Installing inferlib-r2p-echo-callee ==="
pie-client install \
    --path target/wasm32-wasip2/release/inferlib_r2p_echo_callee.wasm \
    --manifest func-call-latency/inferlib-r2p-echo-callee/Pie.toml

echo "=== Installing inferlib-r2r-resecho-callee ==="
pie-client install \
    --path target/wasm32-wasip2/release/inferlib_r2r_resecho_callee.wasm \
    --manifest func-call-latency/inferlib-r2r-resecho-callee/Pie.toml

echo ""
echo "=== Running inferlib cross-component benchmark ==="
echo "=== R2R Echo Dynamic Composition ==="
pie-client submit \
    --path target/wasm32-wasip2/release/inferlib_r2r_echo_caller.wasm \
    --manifest func-call-latency/inferlib-r2r-echo-caller/Pie.toml

echo ""
echo "=== Running inferlib cross-component benchmark ==="
echo "=== R2R Echo Static Composition ==="
pie-client submit \
    --path target/wasm32-wasip2/release/inferlib_r2r_echo_caller.wasm \
    --manifest func-call-latency/inferlib-r2r-echo-caller/Pie.toml \
    --link target/wasm32-wasip2/release/inferlib_r2r_echo_callee.wasm

echo ""
echo "=== Running inferlib cross-component benchmark ==="
echo "=== P2R Echo Dynamic Composition ==="
pie-client submit \
    --path target/wasm32-wasip2/release/inferlib_p2r_echo_caller.wasm \
    --manifest func-call-latency/inferlib-p2r-echo-caller/Pie.toml

echo ""
echo "=== Running inferlib cross-component benchmark ==="
echo "=== P2R Echo Static Composition ==="
pie-client submit \
    --path target/wasm32-wasip2/release/inferlib_p2r_echo_caller.wasm \
    --manifest func-call-latency/inferlib-p2r-echo-caller/Pie.toml \
    --link target/wasm32-wasip2/release/inferlib_p2r_echo_callee.wasm

echo ""
echo "=== Running inferlib cross-component benchmark ==="
echo "=== P2P Echo Dynamic Composition ==="
pie-client submit \
    --path target/wasm32-wasip2/release/inferlib_p2p_echo_caller.wasm \
    --manifest func-call-latency/inferlib-p2p-echo-caller/Pie.toml

echo ""
echo "=== Running inferlib cross-component benchmark ==="
echo "=== P2P Echo Static Composition ==="
pie-client submit \
    --path target/wasm32-wasip2/release/inferlib_p2p_echo_caller.wasm \
    --manifest func-call-latency/inferlib-p2p-echo-caller/Pie.toml \
    --link target/wasm32-wasip2/release/inferlib_p2p_echo_callee.wasm

echo ""
echo "=== Running inferlib cross-component benchmark ==="
echo "=== R2P Echo Dynamic Composition ==="
pie-client submit \
    --path target/wasm32-wasip2/release/inferlib_r2p_echo_caller.wasm \
    --manifest func-call-latency/inferlib-r2p-echo-caller/Pie.toml

echo ""
echo "=== Running inferlib cross-component benchmark ==="
echo "=== R2P Echo Static Composition ==="
pie-client submit \
    --path target/wasm32-wasip2/release/inferlib_r2p_echo_caller.wasm \
    --manifest func-call-latency/inferlib-r2p-echo-caller/Pie.toml \
    --link target/wasm32-wasip2/release/inferlib_r2p_echo_callee.wasm

echo ""
echo "=== Running guest-to-host benchmark (R2H) ==="
pie-client submit \
    --path target/wasm32-wasip2/release/inferlib_r2h_echo_caller.wasm \
    --manifest func-call-latency/inferlib-r2h-echo-caller/Pie.toml

echo ""
echo "=== Running host-to-guest benchmark (H2R) ==="
pie-client submit \
    --path target/wasm32-wasip2/release/inferlib_h2r_echo_caller.wasm \
    --manifest func-call-latency/inferlib-h2r-echo-caller/Pie.toml

for r in 1 2 4 8; do
    echo ""
    echo "=== Running inferlib cross-component benchmark ==="
    echo "=== R2R ResEcho Dynamic Composition Resources=$r ==="
    pie-client submit \
        --path target/wasm32-wasip2/release/inferlib_r2r_resecho_caller.wasm \
        --manifest func-call-latency/inferlib-r2r-resecho-caller/Pie.toml \
        -- -r "$r"
done

for r in 1 2 4 8; do
    echo ""
    echo "=== Running inferlib cross-component benchmark ==="
    echo "=== R2R ResEcho Static Composition Resources=$r ==="
    pie-client submit \
        --path target/wasm32-wasip2/release/inferlib_r2r_resecho_caller.wasm \
        --manifest func-call-latency/inferlib-r2r-resecho-caller/Pie.toml \
        --link target/wasm32-wasip2/release/inferlib_r2r_resecho_callee.wasm \
        -- -r "$r"
done

echo ""
echo "=== Running monolithic intra-component benchmark ==="
pie-client submit \
    --path target/wasm32-wasip2/release/mono_r2r_echo_caller.wasm \
    --manifest func-call-latency/mono-r2r-echo-caller/Pie.toml
