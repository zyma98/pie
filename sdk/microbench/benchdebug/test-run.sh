#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
MICROBENCH_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ENGINE_LOG=$(mktemp)
ENGINE_PID=""

cleanup() {
    if [ -n "$ENGINE_PID" ] && kill -0 "$ENGINE_PID" 2>/dev/null; then
        echo ""
        echo "=== Stopping engine (PID $ENGINE_PID) ==="
        kill -TERM "$ENGINE_PID"
        wait "$ENGINE_PID" 2>/dev/null || true
        echo "Engine stopped."
    fi
    echo "=== Engine Log ==="
    cat "$ENGINE_LOG"
    rm -f "$ENGINE_LOG"
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

echo "=== Building microbench Rust crates ==="
cd "$MICROBENCH_DIR"
cargo build -rq --target wasm32-wasip2 2>/dev/null

CALLEE_SRC="$SCRIPT_DIR/inferlib-debug-r2p-echo-callee"
PY_CALLEE_OUT="$MICROBENCH_DIR/target/wasm32-wasip2/release"
INFERENCE_BINDINGS_WIT="$REPO_ROOT/sdk/rust/inferlib/inference-bindings/wit"

echo "=== Building callee (exec world: run + echo, shared-modules) ==="
APP_TMPDIR=$(mktemp -d)
trap "rm -rf $APP_TMPDIR; $(trap -p EXIT | sed "s/.*'\\(.*\\)'.*/\\1/")" EXIT

APP_WIT_DIR="$APP_TMPDIR/wit"
mkdir -p "$APP_WIT_DIR/deps/echo-callee"
cp -r "$INFERENCE_BINDINGS_WIT/deps"/* "$APP_WIT_DIR/deps/"
cat > "$APP_WIT_DIR/deps/echo-callee/echo.wit" << 'WITEOF'
package microbench:echo-callee;

interface echo {
    echo: func(s: string) -> string;
}
WITEOF

cat > "$APP_WIT_DIR/world.wit" << 'WITEOF'
package pie:inferlib-debug-r2p-echo-callee;

interface run {
    run: func() -> result<_, string>;
}

world exec {
    import wasi:io/poll@0.2.0;
    import inferlib:inference/models;
    import inferlib:inference/queues;
    import inferlib:inference/runtime;
    import inferlib:inference/inference;
    import inferlib:inference/formatter;
    import inferlib:inference/messaging;
    import inferlib:inference/kvstore;

    export run;
    export microbench:echo-callee/echo;
}
WITEOF

cat > "$APP_TMPDIR/app.py" << 'PYEOF'
from wit_world import exports
import inference_bindings as _bindings
import run_bindings as _run_bindings

_return_was_set = False
_raw_set_return = _bindings.set_return

def _tracking_set_return(value: str) -> None:
    global _return_was_set
    _return_was_set = True
    _raw_set_return(value)

_bindings.set_return = _tracking_set_return

import main as _user_module

# Re-export Echo so componentize-py can find it on the entry module
Echo = _user_module.Echo

class Run(exports.Run):
    def run(self) -> None:
        if hasattr(_user_module, 'main'):
            _user_module.main()
        if not _return_was_set:
            _raw_set_return("")
PYEOF

cp "$CALLEE_SRC/main.py" "$APP_TMPDIR/"
cp -r "$CALLEE_SRC/inference_bindings" "$APP_TMPDIR/"
cp -r "$CALLEE_SRC/run_bindings" "$APP_TMPDIR/"

componentize-py -d "$APP_WIT_DIR" -w exec \
    componentize --no-snapshot --shared-modules auto \
    -e "$APP_TMPDIR" \
    -o "$PY_CALLEE_OUT/inferlib_debug_r2p_echo_callee_app.wasm" app

echo "=== Starting engine ==="
cd "$REPO_ROOT/pie"
pie serve > "$ENGINE_LOG" 2>&1 &
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

echo ""
echo "=== Test 1: Submit callee as APPLICATION (should work) ==="
cd "$MICROBENCH_DIR"
pie-client submit \
    --path target/wasm32-wasip2/release/inferlib_debug_r2p_echo_callee_app.wasm \
    --manifest benchdebug/inferlib-debug-r2p-echo-callee/Pie.toml

echo ""
echo "=== Test 2: Install callee as DEPENDENCY, then run caller ==="
pie-client install \
    --path target/wasm32-wasip2/release/inferlib_debug_r2p_echo_callee_app.wasm \
    --manifest benchdebug/inferlib-debug-r2p-echo-callee/Pie.toml

pie-client submit \
    --path target/wasm32-wasip2/release/inferlib_debug_r2p_echo_caller.wasm \
    --manifest benchdebug/inferlib-debug-r2p-echo-caller/Pie.toml

echo ""
echo "=== Done ==="
