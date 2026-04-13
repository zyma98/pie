#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
BUILD_DIR="$SCRIPT_DIR/build"
ENGINE_LOG=$(mktemp)
ENGINE_PID=""
TOXI_PID=""
REPEATS=10
PIE_CACHE_PROGRAMS="$HOME/.pie/cache/programs"
PROXY_PORT=8082

cleanup() {
    if [ -n "$ENGINE_PID" ] && kill -0 "$ENGINE_PID" 2>/dev/null; then
        echo ""
        echo "=== Stopping engine (PID $ENGINE_PID) ==="
        kill -TERM "$ENGINE_PID"
        wait "$ENGINE_PID" 2>/dev/null || true
        echo "Engine stopped."
    fi
    if [ -n "$TOXI_PID" ] && kill -0 "$TOXI_PID" 2>/dev/null; then
        kill -TERM "$TOXI_PID"
        wait "$TOXI_PID" 2>/dev/null || true
    fi
    rm -f "$ENGINE_LOG" "${TEMPLATE_GEN_IMPORTS:-}" "${WATERMARK_IMPORTS:-}"
}
trap cleanup EXIT

mkdir -p "$BUILD_DIR"
source "$REPO_ROOT/pie/.venv/bin/activate"

# ---------------------------------------------------------------------------
# Test cases: display name | wasm path | manifest path | dep_set | herald wasm | herald manifest
#
# dep_set values:
#   none      – no inferlib dependencies
#   inference – inferlib-inference only
#   template  – inferlib-inference + llguidance + schema + template
#
# Herald versions have the same code but a different name. Running the herald
# first causes the engine to compile all shared dependencies, so those costs
# are excluded from the cold-start measurement of the actual application.
# ---------------------------------------------------------------------------
CASES=(
    "rust-mono-template-generation|$REPO_ROOT/sdk/examples/target/wasm32-wasip2/release/template_generation.wasm|$REPO_ROOT/sdk/examples/template-generation/Pie.toml|none|$REPO_ROOT/sdk/examples/target/wasm32-wasip2/release/template_generation_herald.wasm|$REPO_ROOT/sdk/examples/template-generation-herald/Pie.toml"
    "rust-dynamic-template-generation|$REPO_ROOT/sdk/examples-inferlib/target/wasm32-wasip2/release/template_generation.wasm|$REPO_ROOT/sdk/examples-inferlib/template-generation/Pie.toml|template|$REPO_ROOT/sdk/examples-inferlib/target/wasm32-wasip2/release/template_generation_herald.wasm|$REPO_ROOT/sdk/examples-inferlib/template-generation-herald/Pie.toml"
    "rust-static-template-generation|$BUILD_DIR/template-generation-rust-static.wasm|$REPO_ROOT/sdk/examples/template-generation/Pie.toml|none|$REPO_ROOT/sdk/examples/target/wasm32-wasip2/release/template_generation_herald.wasm|$REPO_ROOT/sdk/examples/template-generation-herald/Pie.toml"
    "python-factored-dynamic-template-generation|$BUILD_DIR/template-generation-py.wasm|$REPO_ROOT/sdk/examples-py-inferlib/template-generation/Pie.toml|template|$BUILD_DIR/template-generation-herald-py.wasm|$REPO_ROOT/sdk/examples-py-inferlib/template-generation-herald/Pie.toml"
    "python-full-dynamic-template-generation|$BUILD_DIR/template-generation-full-py.wasm|$REPO_ROOT/sdk/examples-py-inferlib/template-generation-full/Pie.toml|template|$BUILD_DIR/template-generation-herald-py.wasm|$REPO_ROOT/sdk/examples-py-inferlib/template-generation-herald/Pie.toml"
    "python-factored-dynamic-watermarking-numpy|$BUILD_DIR/watermarking-numpy-py.wasm|$REPO_ROOT/sdk/examples-py-inferlib/watermarking-numpy/Pie.toml|inference|$BUILD_DIR/watermarking-numpy-herald-py.wasm|$REPO_ROOT/sdk/examples-py-inferlib/watermarking-numpy-herald/Pie.toml"
    "python-full-dynamic-watermarking-numpy|$BUILD_DIR/watermarking-numpy-full-py.wasm|$REPO_ROOT/sdk/examples-py-inferlib/watermarking-numpy-full/Pie.toml|inference|$BUILD_DIR/watermarking-numpy-herald-py.wasm|$REPO_ROOT/sdk/examples-py-inferlib/watermarking-numpy-herald/Pie.toml"
)

# ---------------------------------------------------------------------------
# Phase 1: Build
# ---------------------------------------------------------------------------

echo "=== Building runtime with case_study_cold_start_latency feature ==="
cd "$REPO_ROOT/pie"
if ! maturin develop --release --features case_study_cold_start_latency > /dev/null 2>&1; then
    echo "maturin build failed! Re-running with output:"
    maturin develop --release --features case_study_cold_start_latency
    exit 1
fi

echo "=== Building inferlib dependencies ==="
cd "$REPO_ROOT/sdk/rust/inferlib"
cargo build -rq --target wasm32-wasip2 2>/dev/null

echo "=== Building sdk/examples (monolithic + herald) ==="
cd "$REPO_ROOT/sdk/examples"
cargo build -rq --target wasm32-wasip2 \
    -p template-generation -p template-generation-herald 2>/dev/null

echo "=== Building sdk/examples-inferlib (rust-inferlib + herald) ==="
cd "$REPO_ROOT/sdk/examples-inferlib"
cargo build -rq --target wasm32-wasip2 \
    -p template-generation -p template-generation-herald 2>/dev/null

IL_WIT="$REPO_ROOT/sdk/rust/inferlib"

# build_py_component <py_src> <pkg_name> <output_wasm> <wit_imports_file> [mode] [extra_pkg_paths]
#   wit_imports_file: path to a file containing the WIT world body (imports + export)
#   mode: "shared" (default) uses --no-snapshot --shared-modules auto -e
#         "full" uses -p (no --no-snapshot, no --shared-modules)
#   extra_pkg_paths: space-separated paths to add as -p to componentize-py
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

    # Copy WIT deps from all bindings referenced by the imports file
    # We always copy inference deps; others only if referenced
    cp -r "$IL_WIT/inference-bindings/wit/deps"/* "$wit_dir/deps/"
    for dep in llguidance schema template; do
        if grep -q "inferlib:${dep}" "$wit_imports_file"; then
            cp -r "$IL_WIT/${dep}-bindings/wit/deps"/* "$wit_dir/deps/"
        fi
    done

    # Generate world.wit
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

    # Copy user source and all *_bindings modules present in the project
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

# WIT imports for template-generation (4 deps)
TEMPLATE_GEN_IMPORTS=$(mktemp)
cat > "$TEMPLATE_GEN_IMPORTS" << 'EOF'
    import wasi:io/poll@0.2.0;
    import inferlib:inference/models;
    import inferlib:inference/queues;
    import inferlib:inference/runtime;
    import inferlib:inference/inference;
    import inferlib:inference/formatter;
    import inferlib:inference/messaging;
    import inferlib:inference/kvstore;
    import inferlib:llguidance/constrained-sampling;
    import inferlib:schema/json-schema;
    import inferlib:template/template-rendering;
EOF

# WIT imports for watermarking-numpy (1 dep: inference only)
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

echo "=== Building Python template-generation ==="
build_py_component \
    "$REPO_ROOT/sdk/examples-py-inferlib/template-generation" \
    "template-generation" \
    "$BUILD_DIR/template-generation-py.wasm" \
    "$TEMPLATE_GEN_IMPORTS"

echo "=== Building Python template-generation-herald ==="
build_py_component \
    "$REPO_ROOT/sdk/examples-py-inferlib/template-generation-herald" \
    "template-generation-herald" \
    "$BUILD_DIR/template-generation-herald-py.wasm" \
    "$TEMPLATE_GEN_IMPORTS"

echo "=== Building Python template-generation-full ==="
build_py_component \
    "$REPO_ROOT/sdk/examples-py-inferlib/template-generation-full" \
    "template-generation-full" \
    "$BUILD_DIR/template-generation-full-py.wasm" \
    "$TEMPLATE_GEN_IMPORTS" \
    full

PY_SITE_PACKAGES="$HOME/.pie/py-runtime/site-packages"

echo "=== Building Python watermarking-numpy ==="
build_py_component \
    "$REPO_ROOT/sdk/examples-py-inferlib/watermarking-numpy" \
    "watermarking-numpy" \
    "$BUILD_DIR/watermarking-numpy-py.wasm" \
    "$WATERMARK_IMPORTS" \
    shared \
    "$PY_SITE_PACKAGES"

echo "=== Building Python watermarking-numpy-herald ==="
build_py_component \
    "$REPO_ROOT/sdk/examples-py-inferlib/watermarking-numpy-herald" \
    "watermarking-numpy-herald" \
    "$BUILD_DIR/watermarking-numpy-herald-py.wasm" \
    "$WATERMARK_IMPORTS" \
    shared \
    "$PY_SITE_PACKAGES"

echo "=== Building Python watermarking-numpy-full ==="
build_py_component \
    "$REPO_ROOT/sdk/examples-py-inferlib/watermarking-numpy-full" \
    "watermarking-numpy-full" \
    "$BUILD_DIR/watermarking-numpy-full-py.wasm" \
    "$WATERMARK_IMPORTS" \
    full \
    "$PY_SITE_PACKAGES"

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

echo "=== Pre-linking rust-inferlib-static ==="
wac_plug "$REPO_ROOT/sdk/examples-inferlib/target/wasm32-wasip2/release/template_generation.wasm" \
    "$BUILD_DIR/template-generation-rust-static.wasm" \
    "$_IL/inferlib_inference.wasm" \
    "$_IL/inferlib_llguidance.wasm" \
    "$_IL/inferlib_schema.wasm" \
    "$_IL/inferlib_template.wasm"

# ---------------------------------------------------------------------------
# Toxiproxy setup
# ---------------------------------------------------------------------------

echo "=== Setting up toxiproxy ==="
toxiproxy-server &>/dev/null &
TOXI_PID=$!
sleep 1
toxiproxy-cli delete pie_engine >/dev/null 2>&1 || true
toxiproxy-cli create --listen "127.0.0.1:${PROXY_PORT}" --upstream 127.0.0.1:8080 pie_engine >/dev/null

# Network bandwidth scenarios: name | rate in KB/s
#   10 Mbps  ≈  1250 KB/s
#  100 Mbps  ≈ 12500 KB/s
# 1000 Mbps  ≈ 125000 KB/s
BANDWIDTHS=(
    "slow|1250"
    "medium|12500"
    "fast|125000"
)

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

install_deps() {
    local dep_set="$1"
    case "$dep_set" in
        none)
            ;;
        inference)
            pie-client install \
                --path "$REPO_ROOT/sdk/rust/inferlib/target/wasm32-wasip2/release/inferlib_inference.wasm" \
                --manifest "$REPO_ROOT/sdk/rust/inferlib/inference/Pie.toml" > /dev/null 2>&1
            ;;
        template)
            for dep in inference llguidance schema template; do
                pie-client install \
                    --path "$REPO_ROOT/sdk/rust/inferlib/target/wasm32-wasip2/release/inferlib_${dep}.wasm" \
                    --manifest "$REPO_ROOT/sdk/rust/inferlib/$dep/Pie.toml" > /dev/null 2>&1
            done
            ;;
        *)
            echo "Unknown dep_set: $dep_set" >&2
            exit 1
            ;;
    esac
}

submit_app() {
    local wasm_path="$1"
    local manifest_path="$2"
    local port_args=""
    if [ -n "${3:-}" ]; then
        port_args="--port $3"
    fi
    # shellcheck disable=SC2086
    pie-client submit $port_args \
        --path "$wasm_path" \
        --manifest "$manifest_path" > /dev/null 2>&1
}

set_bandwidth() {
    local rate="$1"
    toxiproxy-cli toxic remove --toxicName bandwidth_downstream pie_engine >/dev/null 2>&1 || true
    toxiproxy-cli toxic remove --toxicName bandwidth_upstream pie_engine >/dev/null 2>&1 || true
    toxiproxy-cli toxic add --type bandwidth --attribute "rate=$rate" pie_engine >/dev/null
    toxiproxy-cli toxic add --type bandwidth --attribute "rate=$rate" --upstream pie_engine >/dev/null
}

extract_cold_start_us() {
    grep -a "\[case-study\].*Cold-start latency" "$ENGINE_LOG" | \
        tail -1 | \
        sed "s/.*Cold-start latency: *\([0-9.]*\) us/\1/"
}

extract_upload_us() {
    grep -a "\[case-study\].*Cold-start upload" "$ENGINE_LOG" | \
        tail -1 | \
        sed "s/.*Cold-start upload: *\([0-9.]*\) us/\1/"
}

extract_compile_us() {
    grep -a "\[case-study\].*Cold-start compile" "$ENGINE_LOG" | \
        tail -1 | \
        sed "s/.*Cold-start compile: *\([0-9.]*\) us/\1/"
}

# ---------------------------------------------------------------------------
# Phase 2: Run each case REPEATS times, for each bandwidth scenario
# ---------------------------------------------------------------------------

declare -A all_cold_start
declare -A all_upload
declare -A all_compile

for bw_entry in "${BANDWIDTHS[@]}"; do
    IFS='|' read -r bw_name bw_rate <<< "$bw_entry"

    echo ""
    echo "############################################################"
    echo "  Network: ${bw_name} (~$(( bw_rate * 8 / 1000 )) Mbps, ${bw_rate} KB/s)"
    echo "############################################################"

    set_bandwidth "$bw_rate"

    for entry in "${CASES[@]}"; do
        IFS='|' read -r name wasm_path manifest_path dep_set herald_wasm herald_manifest <<< "$entry"
        key="${bw_name}/${name}"

        echo ""
        echo "============================================================"
        echo "  ${bw_name} / ${name} (${REPEATS} runs)"
        echo "============================================================"

        case_cs=""
        case_up=""
        case_cp=""

        for i in $(seq 1 $REPEATS); do
            echo -n "  Run $i/$REPEATS: clearing cache..."
            rm -rf "$PIE_CACHE_PROGRAMS"

            echo -n " starting engine..."
            start_engine

            echo -n " installing deps..."
            install_deps "$dep_set"

            echo -n " warming up (herald)..."
            submit_app "$herald_wasm" "$herald_manifest"

            echo -n " submitting (throttled)..."
            submit_app "$wasm_path" "$manifest_path" "$PROXY_PORT"

            cs=$(extract_cold_start_us)
            up=$(extract_upload_us)
            cp=$(extract_compile_us)

            echo " upload=${up} us  compile=${cp} us  total=${cs} us"

            case_cs="$case_cs $cs"
            case_up="$case_up $up"
            case_cp="$case_cp $cp"

            stop_engine
        done

        all_cold_start["$key"]="$case_cs"
        all_upload["$key"]="$case_up"
        all_compile["$key"]="$case_cp"

        python3 -c "
cs = [float(x) for x in '''$case_cs'''.split()]
up = [float(x) for x in '''$case_up'''.split()]
cp = [float(x) for x in '''$case_cp'''.split()]
print()
print('  --- Average ---')
print(f'  Upload:              {sum(up)/len(up)/1e6:.3f} s')
print(f'  Compile+instantiate: {sum(cp)/len(cp)/1e6:.3f} s')
print(f'  Cold-start latency:  {sum(cs)/len(cs)/1e6:.3f} s')
"
    done
done

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

echo ""
echo "============================================================"
echo "  Overall Summary ($REPEATS runs each)"
echo "============================================================"

echo ""
echo "--- Wasm file sizes ---"
for entry in "${CASES[@]}"; do
    IFS='|' read -r name wasm_path _ _ _ _ <<< "$entry"
    size_bytes=$(stat --format=%s "$wasm_path")
    size_mb=$(python3 -c "print(f'{$size_bytes / 1048576:.2f}')")
    echo "  ${name}:  ${size_mb} MB"
done

for bw_entry in "${BANDWIDTHS[@]}"; do
    IFS='|' read -r bw_name bw_rate <<< "$bw_entry"
    echo ""
    echo "--- ${bw_name} (~$(( bw_rate * 8 / 1000 )) Mbps) ---"
    for entry in "${CASES[@]}"; do
        IFS='|' read -r name _ _ _ _ _ <<< "$entry"
        key="${bw_name}/${name}"
        python3 -c "
cs = [float(x) for x in '''${all_cold_start[$key]}'''.split()]
up = [float(x) for x in '''${all_upload[$key]}'''.split()]
cp = [float(x) for x in '''${all_compile[$key]}'''.split()]
print(f'  ${name}:  upload={sum(up)/len(up)/1e6:.3f} s  compile={sum(cp)/len(cp)/1e6:.3f} s  total={sum(cs)/len(cs)/1e6:.3f} s')
"
    done
done
