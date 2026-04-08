#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
BUILD_DIR="$SCRIPT_DIR/build"
ENGINE_LOG=$(mktemp)
ENGINE_PID=""
NUM_INSTANCES_RUST=1600
NUM_INSTANCES_PYTHON=700
NUM_TICK_RUST=80
NUM_TICK_PYTHON=35
PIE_CACHE_PROGRAMS="$HOME/.pie/cache/programs"

cleanup() {
    if [ -n "$ENGINE_PID" ] && kill -0 "$ENGINE_PID" 2>/dev/null; then
        echo ""
        echo "=== Stopping engine (PID $ENGINE_PID) ==="
        kill -TERM "$ENGINE_PID"
        wait "$ENGINE_PID" 2>/dev/null || true
        echo "Engine stopped."
    fi
    rm -f "$ENGINE_LOG" "${WATERMARK_IMPORTS:-}"
}
trap cleanup EXIT

mkdir -p "$BUILD_DIR"
source "$REPO_ROOT/pie/.venv/bin/activate"

# ---------------------------------------------------------------------------
# Phase 1: Build
# ---------------------------------------------------------------------------

echo "=== Building runtime with case_study_memory_consumption feature ==="
cd "$REPO_ROOT/pie"
if ! maturin develop --release --features case_study_memory_consumption > /dev/null 2>&1; then
    echo "maturin build failed! Re-running with output:"
    maturin develop --release --features case_study_memory_consumption
    exit 1
fi

echo "=== Building inferlib dependencies ==="
cd "$REPO_ROOT/sdk/rust/inferlib"
cargo build -rq --target wasm32-wasip2 2>/dev/null

echo "=== Building sdk/examples (monolithic template-generation-sleep) ==="
cd "$REPO_ROOT/sdk/examples"
cargo build -rq --target wasm32-wasip2 -p template-generation-sleep 2>/dev/null

echo "=== Building sdk/examples-inferlib (dynamic template-generation-sleep) ==="
cd "$REPO_ROOT/sdk/examples-inferlib"
cargo build -rq --target wasm32-wasip2 -p template-generation-sleep 2>/dev/null

# Pre-link static binary using wac plug
wac_plug() {
    local src="$1" out="$2"
    shift 2
    local current="$src" tmp
    for lib in "$@"; do
        tmp=$(mktemp --suffix=.wasm)
        wac plug --plug "$lib" "$current" -o "$tmp"
        if [ "$current" != "$src" ]; then rm -f "$current"; fi
        current="$tmp"
    done
    mv "$current" "$out"
}

_IL="$REPO_ROOT/sdk/rust/inferlib/target/wasm32-wasip2/release"

echo "=== Pre-linking rust-static template-generation-sleep ==="
wac_plug "$REPO_ROOT/sdk/examples-inferlib/target/wasm32-wasip2/release/template_generation_sleep.wasm" \
    "$BUILD_DIR/template-generation-sleep-static.wasm" \
    "$_IL/inferlib_inference.wasm" \
    "$_IL/inferlib_llguidance.wasm" \
    "$_IL/inferlib_schema.wasm" \
    "$_IL/inferlib_template.wasm"

IL_WIT="$REPO_ROOT/sdk/rust/inferlib"

# build_py_component <py_src> <pkg_name> <output_wasm> <wit_imports_file> [mode] [extra_pkg_paths]
build_py_component() {
    local py_src="$1"
    local pkg_name="$2"
    local output_wasm="$3"
    local wit_imports_file="$4"
    local mode="${5:-shared}"
    local extra_pkg_paths="${6:-}"

    local tmpdir
    tmpdir=$(mktemp -d)

    local wit_dir="$tmpdir/wit"
    mkdir -p "$wit_dir/deps"

    cp -r "$IL_WIT/inference-bindings/wit/deps"/* "$wit_dir/deps/"
    for dep in llguidance schema template; do
        if grep -q "inferlib:${dep}" "$wit_imports_file"; then
            cp -r "$IL_WIT/${dep}-bindings/wit/deps"/* "$wit_dir/deps/"
        fi
    done

    {
        echo "package pie:${pkg_name};"
        echo ""
        echo "interface run {"
        echo "    run: func() -> result<_, string>;"
        echo "}"
        echo ""
        echo "world exec {"
        cat "$wit_imports_file"
        echo ""
        echo "    export run;"
        echo "}"
    } > "$wit_dir/world.wit"

    cat > "$tmpdir/app.py" << 'PYEOF'
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

class Run(exports.Run):
    def run(self) -> None:
        if hasattr(_user_module, 'main'):
            _user_module.main()
        if not _return_was_set:
            _raw_set_return("")
PYEOF

    cp "$py_src/main.py" "$tmpdir/"
    for mod_dir in "$py_src"/*_bindings; do
        [ -d "$mod_dir" ] && cp -r "$mod_dir" "$tmpdir/"
    done

    local extra_p_args=""
    for p in $extra_pkg_paths; do
        extra_p_args="$extra_p_args -p $p"
    done

    if [ "$mode" = "full" ]; then
        # shellcheck disable=SC2086
        componentize-py -d "$wit_dir" -w exec \
            componentize \
            -p "$tmpdir" $extra_p_args \
            -o "$output_wasm" app 2>/dev/null
    else
        # shellcheck disable=SC2086
        componentize-py -d "$wit_dir" -w exec \
            componentize --no-snapshot --shared-modules auto \
            -e "$tmpdir" $extra_p_args \
            -o "$output_wasm" app 2>/dev/null
    fi

    rm -rf "$tmpdir"
}

WATERMARK_IMPORTS=$(mktemp)
cat > "$WATERMARK_IMPORTS" << 'EOF'
    import wasi:io/poll@0.2.0;
    import inferlib:inference/models;
    import inferlib:inference/queues;
    import inferlib:inference/runtime;
    import inferlib:inference/inference;
    import inferlib:inference/formatter;
    import inferlib:inference/messaging;
    import inferlib:inference/kvstore;
EOF

PY_SITE_PACKAGES="$HOME/.pie/py-runtime/site-packages"

echo "=== Building Python watermarking-numpy-full-sleep ==="
build_py_component \
    "$REPO_ROOT/sdk/examples-py-inferlib/watermarking-numpy-full-sleep" \
    "watermarking-numpy-full-sleep" \
    "$BUILD_DIR/watermarking-numpy-full-sleep.wasm" \
    "$WATERMARK_IMPORTS" \
    full \
    "$PY_SITE_PACKAGES"

echo "=== Building Python watermarking-numpy-sleep ==="
build_py_component \
    "$REPO_ROOT/sdk/examples-py-inferlib/watermarking-numpy-sleep" \
    "watermarking-numpy-sleep" \
    "$BUILD_DIR/watermarking-numpy-sleep.wasm" \
    "$WATERMARK_IMPORTS" \
    shared \
    "$PY_SITE_PACKAGES"

# Generate manifest copies with unique descriptions
MANIFESTS_FULL_DIR="$BUILD_DIR/manifests-full"
rm -rf "$MANIFESTS_FULL_DIR"
mkdir -p "$MANIFESTS_FULL_DIR"

echo "=== Generating $NUM_INSTANCES_PYTHON manifest copies (full) ==="
for i in $(seq 1 $NUM_INSTANCES_PYTHON); do
    sed "s/description = \"/description = \"(Case ${i}) /" \
        "$REPO_ROOT/sdk/examples-py-inferlib/watermarking-numpy-full-sleep/Pie.toml" \
        > "$MANIFESTS_FULL_DIR/Pie-${i}.toml"
done

MANIFESTS_SHARED_DIR="$BUILD_DIR/manifests-shared"
rm -rf "$MANIFESTS_SHARED_DIR"
mkdir -p "$MANIFESTS_SHARED_DIR"

echo "=== Generating $NUM_INSTANCES_PYTHON manifest copies (shared) ==="
for i in $(seq 1 $NUM_INSTANCES_PYTHON); do
    sed "s/description = \"/description = \"(Case ${i}) /" \
        "$REPO_ROOT/sdk/examples-py-inferlib/watermarking-numpy-sleep/Pie.toml" \
        > "$MANIFESTS_SHARED_DIR/Pie-${i}.toml"
done

MANIFESTS_RUST_MONO_DIR="$BUILD_DIR/manifests-rust-mono"
rm -rf "$MANIFESTS_RUST_MONO_DIR"
mkdir -p "$MANIFESTS_RUST_MONO_DIR"

echo "=== Generating $NUM_INSTANCES_RUST manifest copies (rust monolithic) ==="
for i in $(seq 1 $NUM_INSTANCES_RUST); do
    sed "s/description = \"/description = \"(Case ${i}) /" \
        "$REPO_ROOT/sdk/examples/template-generation-sleep/Pie.toml" \
        > "$MANIFESTS_RUST_MONO_DIR/Pie-${i}.toml"
done

MANIFESTS_RUST_DYN_DIR="$BUILD_DIR/manifests-rust-dynamic"
rm -rf "$MANIFESTS_RUST_DYN_DIR"
mkdir -p "$MANIFESTS_RUST_DYN_DIR"

echo "=== Generating $NUM_INSTANCES_RUST manifest copies (rust dynamic) ==="
for i in $(seq 1 $NUM_INSTANCES_RUST); do
    sed "s/description = \"/description = \"(Case ${i}) /" \
        "$REPO_ROOT/sdk/examples-inferlib/template-generation-sleep/Pie.toml" \
        > "$MANIFESTS_RUST_DYN_DIR/Pie-${i}.toml"
done

# Static uses the monolithic manifest (no runtime deps since pre-linked)
MANIFESTS_RUST_STATIC_DIR="$BUILD_DIR/manifests-rust-static"
rm -rf "$MANIFESTS_RUST_STATIC_DIR"
mkdir -p "$MANIFESTS_RUST_STATIC_DIR"

echo "=== Generating $NUM_INSTANCES_RUST manifest copies (rust static) ==="
for i in $(seq 1 $NUM_INSTANCES_RUST); do
    sed "s/description = \"/description = \"(Case ${i}) /" \
        "$REPO_ROOT/sdk/examples/template-generation-sleep/Pie.toml" \
        > "$MANIFESTS_RUST_STATIC_DIR/Pie-${i}.toml"
done

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

start_engine() {
    : > "$ENGINE_LOG"
    cd "$REPO_ROOT/pie"
    pie serve > "$ENGINE_LOG" 2>&1 &
    ENGINE_PID=$!

    while ! grep -q "Engine running" "$ENGINE_LOG" 2>/dev/null; do
        if ! kill -0 "$ENGINE_PID" 2>/dev/null; then
            echo " FAILED"
            echo "Engine exited unexpectedly. Log output:"
            cat "$ENGINE_LOG"
            exit 1
        fi
        sleep 1
    done
}

stop_engine() {
    if [ -n "$ENGINE_PID" ] && kill -0 "$ENGINE_PID" 2>/dev/null; then
        kill -TERM "$ENGINE_PID"
        wait "$ENGINE_PID" 2>/dev/null || true
        ENGINE_PID=""
    fi
}

get_tree_pids() {
    local pid="$1"
    echo "$pid"
    local children
    children=$(pgrep -P "$pid" 2>/dev/null) || true
    for child in $children; do
        get_tree_pids "$child"
    done
}

measure_rss_kb() {
    local total=0
    for pid in $(get_tree_pids "$ENGINE_PID"); do
        local rss
        rss=$(ps -o rss= -p "$pid" 2>/dev/null | tr -d ' ') || true
        if [ -n "$rss" ]; then
            total=$((total + rss))
        fi
    done
    echo "$total"
}

print_process_tree() {
    for pid in $(get_tree_pids "$ENGINE_PID"); do
        rss=$(ps -o rss= -p "$pid" 2>/dev/null | tr -d ' ') || true
        cmd=$(ps -o args= -p "$pid" 2>/dev/null) || true
        printf "  PID %-8s  RSS %8s KB  %s\n" "$pid" "${rss:-0}" "${cmd:-<unknown>}"
    done
}

print_memory_summary() {
    local label="$1"
    local total_kb
    total_kb=$(measure_rss_kb)
    local total_mib
    total_mib=$(python3 -c "print(f'{${total_kb} / 1024:.2f}')")
    echo "=== ${label}: ${total_kb} KB (${total_mib} MiB) ==="
}

install_inference_dep() {
    pie-client install \
        --path "$REPO_ROOT/sdk/rust/inferlib/target/wasm32-wasip2/release/inferlib_inference.wasm" \
        --manifest "$REPO_ROOT/sdk/rust/inferlib/inference/Pie.toml" > /dev/null 2>&1
}

install_template_deps() {
    for dep in inference llguidance schema template; do
        pie-client install \
            --path "$REPO_ROOT/sdk/rust/inferlib/target/wasm32-wasip2/release/inferlib_${dep}.wasm" \
            --manifest "$REPO_ROOT/sdk/rust/inferlib/$dep/Pie.toml" > /dev/null 2>&1
    done
}

verify_detached_count() {
    local expected="$1"
    local count
    count=$(pie-client list 2>/dev/null | grep -c "Detached" || true)
    if [ "$count" -eq "$expected" ]; then
        echo "  Verified: $count detached instances (expected $expected)"
    else
        echo "  WARNING: found $count detached instances, expected $expected"
    fi
}

# ---------------------------------------------------------------------------
# Case 1: Idle engine memory
# ---------------------------------------------------------------------------

echo ""
echo "============================================================"
echo "  Case 1: Idle engine"
echo "============================================================"

echo -n "Clearing cache..."
rm -rf "$PIE_CACHE_PROGRAMS"
echo -n " starting engine..."
start_engine
echo " ready (PID $ENGINE_PID)."

sleep 2

echo ""
echo "--- Process tree ---"
print_process_tree
echo ""
print_memory_summary "Idle engine RSS"

stop_engine

# ---------------------------------------------------------------------------
# Case 2: Engine with NUM_INSTANCES_PYTHON detached inferlets (full mode)
# ---------------------------------------------------------------------------

echo ""
echo "============================================================"
echo "  Case 2: ${NUM_INSTANCES_PYTHON} detached inferlets (python full)"
echo "============================================================"

echo -n "Clearing cache..."
rm -rf "$PIE_CACHE_PROGRAMS"
echo -n " starting engine..."
start_engine
echo " ready (PID $ENGINE_PID)."
echo "--- Process tree (sanity check) ---"
print_process_tree

echo -n "Installing inferlib-inference dependency..."
install_inference_dep
echo " done."

echo "Submitting ${NUM_INSTANCES_PYTHON} detached inferlets (full)..."
for i in $(seq 1 $NUM_INSTANCES_PYTHON); do
    pie-client submit --detached \
        --path "$BUILD_DIR/watermarking-numpy-full-sleep.wasm" \
        --manifest "$MANIFESTS_FULL_DIR/Pie-${i}.toml" > /dev/null 2>&1
    if (( i % NUM_TICK_PYTHON == 0 )); then
        rss_kb=$(measure_rss_kb)
        echo "  submitted $i/$NUM_INSTANCES_PYTHON  RSS=${rss_kb} KB"
    fi
done
echo "All ${NUM_INSTANCES_PYTHON} submissions complete."
verify_detached_count "$NUM_INSTANCES_PYTHON"

stop_engine

# ---------------------------------------------------------------------------
# Case 3: Engine with NUM_INSTANCES_PYTHON detached inferlets (shared mode)
# ---------------------------------------------------------------------------

echo ""
echo "============================================================"
echo "  Case 3: ${NUM_INSTANCES_PYTHON} detached inferlets (python shared)"
echo "============================================================"

echo -n "Clearing cache..."
rm -rf "$PIE_CACHE_PROGRAMS"
echo -n " starting engine..."
start_engine
echo " ready (PID $ENGINE_PID)."
echo "--- Process tree (sanity check) ---"
print_process_tree

echo -n "Installing inferlib-inference dependency..."
install_inference_dep
echo " done."

echo "Submitting ${NUM_INSTANCES_PYTHON} detached inferlets (shared)..."
for i in $(seq 1 $NUM_INSTANCES_PYTHON); do
    pie-client submit --detached \
        --path "$BUILD_DIR/watermarking-numpy-sleep.wasm" \
        --manifest "$MANIFESTS_SHARED_DIR/Pie-${i}.toml" > /dev/null 2>&1
    if (( i % NUM_TICK_PYTHON == 0 )); then
        rss_kb=$(measure_rss_kb)
        echo "  submitted $i/$NUM_INSTANCES_PYTHON  RSS=${rss_kb} KB"
    fi
done
echo "All ${NUM_INSTANCES_PYTHON} submissions complete."
verify_detached_count "$NUM_INSTANCES_PYTHON"

stop_engine

# ---------------------------------------------------------------------------
# Case 4: Rust monolithic template-generation-sleep
# ---------------------------------------------------------------------------

echo ""
echo "============================================================"
echo "  Case 4: ${NUM_INSTANCES_RUST} detached inferlets (rust monolithic)"
echo "============================================================"

echo -n "Clearing cache..."
rm -rf "$PIE_CACHE_PROGRAMS"
echo -n " starting engine..."
start_engine
echo " ready (PID $ENGINE_PID)."
echo "--- Process tree (sanity check) ---"
print_process_tree

RUST_MONO_WASM="$REPO_ROOT/sdk/examples/target/wasm32-wasip2/release/template_generation_sleep.wasm"

echo "Submitting ${NUM_INSTANCES_RUST} detached inferlets (rust monolithic)..."
for i in $(seq 1 $NUM_INSTANCES_RUST); do
    pie-client submit --detached \
        --path "$RUST_MONO_WASM" \
        --manifest "$MANIFESTS_RUST_MONO_DIR/Pie-${i}.toml" > /dev/null 2>&1
    if (( i % NUM_TICK_RUST == 0 )); then
        rss_kb=$(measure_rss_kb)
        echo "  submitted $i/$NUM_INSTANCES_RUST  RSS=${rss_kb} KB"
    fi
done
echo "All ${NUM_INSTANCES_RUST} submissions complete."
verify_detached_count "$NUM_INSTANCES_RUST"

stop_engine

# ---------------------------------------------------------------------------
# Case 5: Rust dynamic template-generation-sleep
# ---------------------------------------------------------------------------

echo ""
echo "============================================================"
echo "  Case 5: ${NUM_INSTANCES_RUST} detached inferlets (rust dynamic)"
echo "============================================================"

echo -n "Clearing cache..."
rm -rf "$PIE_CACHE_PROGRAMS"
echo -n " starting engine..."
start_engine
echo " ready (PID $ENGINE_PID)."
echo "--- Process tree (sanity check) ---"
print_process_tree

echo -n "Installing inferlib template dependencies..."
install_template_deps
echo " done."

RUST_DYN_WASM="$REPO_ROOT/sdk/examples-inferlib/target/wasm32-wasip2/release/template_generation_sleep.wasm"

echo "Submitting ${NUM_INSTANCES_RUST} detached inferlets (rust dynamic)..."
for i in $(seq 1 $NUM_INSTANCES_RUST); do
    pie-client submit --detached \
        --path "$RUST_DYN_WASM" \
        --manifest "$MANIFESTS_RUST_DYN_DIR/Pie-${i}.toml" > /dev/null 2>&1
    if (( i % NUM_TICK_RUST == 0 )); then
        rss_kb=$(measure_rss_kb)
        echo "  submitted $i/$NUM_INSTANCES_RUST  RSS=${rss_kb} KB"
    fi
done
echo "All ${NUM_INSTANCES_RUST} submissions complete."
verify_detached_count "$NUM_INSTANCES_RUST"

stop_engine

# ---------------------------------------------------------------------------
# Case 6: Rust static template-generation-sleep
# ---------------------------------------------------------------------------

echo ""
echo "============================================================"
echo "  Case 6: ${NUM_INSTANCES_RUST} detached inferlets (rust static)"
echo "============================================================"

echo -n "Clearing cache..."
rm -rf "$PIE_CACHE_PROGRAMS"
echo -n " starting engine..."
start_engine
echo " ready (PID $ENGINE_PID)."
echo "--- Process tree (sanity check) ---"
print_process_tree

echo "Submitting ${NUM_INSTANCES_RUST} detached inferlets (rust static)..."
for i in $(seq 1 $NUM_INSTANCES_RUST); do
    pie-client submit --detached \
        --path "$BUILD_DIR/template-generation-sleep-static.wasm" \
        --manifest "$MANIFESTS_RUST_STATIC_DIR/Pie-${i}.toml" > /dev/null 2>&1
    if (( i % NUM_TICK_RUST == 0 )); then
        rss_kb=$(measure_rss_kb)
        echo "  submitted $i/$NUM_INSTANCES_RUST  RSS=${rss_kb} KB"
    fi
done
echo "All ${NUM_INSTANCES_RUST} submissions complete."
verify_detached_count "$NUM_INSTANCES_RUST"

stop_engine
echo ""
echo "Done."
