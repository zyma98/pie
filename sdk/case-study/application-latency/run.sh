#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
BUILD_DIR="$SCRIPT_DIR/build"
ENGINE_LOG=$(mktemp)
ENGINE_PID=""
REPEATS=10

cleanup() {
    if [ -n "$ENGINE_PID" ] && kill -0 "$ENGINE_PID" 2>/dev/null; then
        echo ""
        echo "=== Stopping engine (PID $ENGINE_PID) ==="
        kill -TERM "$ENGINE_PID"
        wait "$ENGINE_PID" 2>/dev/null || true
    fi
    rm -f "$ENGINE_LOG"
}
trap cleanup EXIT

mkdir -p "$BUILD_DIR"
source "$REPO_ROOT/pie/.venv/bin/activate"

# ---------------------------------------------------------------------------
# Test cases: display name | absolute wasm path | absolute manifest path | dep_set | extra submit args
#
# dep_set values:
#   none      – no inferlib dependencies
#   inference – inferlib-inference only
#   template  – inferlib-inference + llguidance + schema + template
#   cacheback – inferlib-inference + cacheback
#   codeact   – inferlib-inference + js-engine
#
# *-static cases use pre-linked WASM binaries built in Phase 1 (dep_set=none).
# ---------------------------------------------------------------------------
CASES=(
    "prefix-tree/monolithic|$REPO_ROOT/sdk/examples/target/wasm32-wasip2/release/prefix_tree.wasm|$REPO_ROOT/sdk/examples/prefix-tree/Pie.toml|none|-- --num-tokens 128"
    "prefix-tree/rust-inferlib|$REPO_ROOT/sdk/examples-inferlib/target/wasm32-wasip2/release/prefix_tree.wasm|$REPO_ROOT/sdk/examples-inferlib/prefix-tree/Pie.toml|inference|-- --num-tokens 128"
    "prefix-tree/rust-inferlib-static|$BUILD_DIR/prefix-tree-static.wasm|$REPO_ROOT/sdk/examples-inferlib/prefix-tree/Pie.toml|none|-- --num-tokens 128"
    "prefix-tree/python-inferlib|$BUILD_DIR/prefix-tree-py.wasm|$REPO_ROOT/sdk/examples-py-inferlib/prefix-tree/Pie.toml|inference|-- --num-tokens 128"
    "rot/monolithic|$REPO_ROOT/sdk/examples/target/wasm32-wasip2/release/recursion_of_thought.wasm|$REPO_ROOT/sdk/examples/recursion-of-thought/Pie.toml|none|-- --max-tokens 4096"
    "rot/rust-inferlib|$REPO_ROOT/sdk/examples-inferlib/target/wasm32-wasip2/release/recursion_of_thought.wasm|$REPO_ROOT/sdk/examples-inferlib/recursion-of-thought/Pie.toml|inference|-- --max-tokens 4096"
    "rot/rust-inferlib-static|$BUILD_DIR/rot-static.wasm|$REPO_ROOT/sdk/examples-inferlib/recursion-of-thought/Pie.toml|none|-- --max-tokens 4096"
    "rot/python-inferlib|$BUILD_DIR/recursion-of-thought-py.wasm|$REPO_ROOT/sdk/examples-py-inferlib/recursion-of-thought/Pie.toml|inference|-- --max-tokens 4096"
    "template-gen/monolithic|$REPO_ROOT/sdk/examples/target/wasm32-wasip2/release/template_generation.wasm|$REPO_ROOT/sdk/examples/template-generation/Pie.toml|none|"
    "template-gen/rust-inferlib|$REPO_ROOT/sdk/examples-inferlib/target/wasm32-wasip2/release/template_generation.wasm|$REPO_ROOT/sdk/examples-inferlib/template-generation/Pie.toml|template|"
    "template-gen/rust-inferlib-static|$BUILD_DIR/template-gen-static.wasm|$REPO_ROOT/sdk/examples-inferlib/template-generation/Pie.toml|none|"
    "template-gen/python-inferlib|$BUILD_DIR/template-generation-py.wasm|$REPO_ROOT/sdk/examples-py-inferlib/template-generation/Pie.toml|template|"
    "cacheback/monolithic|$REPO_ROOT/sdk/examples/target/wasm32-wasip2/release/cacheback_decoding.wasm|$REPO_ROOT/sdk/examples/cacheback-decoding/Pie.toml|none|-- --max-tokens 512"
    "cacheback/rust-inferlib|$REPO_ROOT/sdk/examples-inferlib/target/wasm32-wasip2/release/cacheback_decoding.wasm|$REPO_ROOT/sdk/examples-inferlib/cacheback-decoding/Pie.toml|cacheback|-- --max-tokens 512"
    "cacheback/rust-inferlib-static|$BUILD_DIR/cacheback-static.wasm|$REPO_ROOT/sdk/examples-inferlib/cacheback-decoding/Pie.toml|none|-- --max-tokens 512"
    "cacheback/python-inferlib|$BUILD_DIR/cacheback-decoding-py.wasm|$REPO_ROOT/sdk/examples-py-inferlib/cacheback-decoding/Pie.toml|cacheback|-- --max-tokens 512"
    "codeact/monolithic|$REPO_ROOT/sdk/examples/target/wasm32-wasip2/release/agent_codeact.wasm|$REPO_ROOT/sdk/examples/agent-codeact/Pie.toml|none|-- --tokens-between-calls 2048"
    "codeact/rust-inferlib|$REPO_ROOT/sdk/examples-inferlib/target/wasm32-wasip2/release/agent_codeact.wasm|$REPO_ROOT/sdk/examples-inferlib/agent-codeact/Pie.toml|codeact|-- --tokens-between-calls 2048"
    "codeact/rust-inferlib-static|$BUILD_DIR/codeact-static.wasm|$REPO_ROOT/sdk/examples-inferlib/agent-codeact/Pie.toml|none|-- --tokens-between-calls 2048"
    "codeact/python-inferlib|$BUILD_DIR/agent-codeact-py.wasm|$REPO_ROOT/sdk/examples-py-inferlib/agent-codeact/Pie.toml|codeact|-- --tokens-between-calls 2048"
    "text-completion/monolithic|$REPO_ROOT/sdk/examples/target/wasm32-wasip2/release/text_completion_greedy.wasm|$REPO_ROOT/sdk/examples/text-completion-greedy/Pie.toml|none|-- --max-tokens 512"
    "text-completion/rust-inferlib|$REPO_ROOT/sdk/examples-inferlib/target/wasm32-wasip2/release/text_completion_greedy.wasm|$REPO_ROOT/sdk/examples-inferlib/text-completion-greedy/Pie.toml|inference|-- --max-tokens 512"
    "text-completion/rust-inferlib-static|$BUILD_DIR/text-completion-static.wasm|$REPO_ROOT/sdk/examples-inferlib/text-completion-greedy/Pie.toml|none|-- --max-tokens 512"
    "text-completion/python-inferlib|$BUILD_DIR/text-completion-greedy-py.wasm|$REPO_ROOT/sdk/examples-py-inferlib/text-completion-greedy/Pie.toml|inference|-- --max-tokens 512"
)

# ---------------------------------------------------------------------------
# Phase 1: Build
# ---------------------------------------------------------------------------

echo "=== Building sdk/examples ==="
cd "$REPO_ROOT/sdk/examples"
cargo build -rq --target wasm32-wasip2 \
    -p prefix-tree -p recursion-of-thought -p template-generation \
    -p cacheback-decoding -p agent-codeact -p text-completion-greedy 2>/dev/null

echo "=== Building sdk/examples-inferlib ==="
cd "$REPO_ROOT/sdk/examples-inferlib"
cargo build -rq --target wasm32-wasip2 \
    -p prefix-tree -p recursion-of-thought -p template-generation \
    -p cacheback-decoding -p agent-codeact -p text-completion-greedy 2>/dev/null

echo "=== Building Python prefix-tree ==="
bakery build \
    --inferlib "$REPO_ROOT/sdk/rust/inferlib" \
    --bindings "$REPO_ROOT/sdk/rust/inferlib/py-bindings" \
    "$REPO_ROOT/sdk/examples-py-inferlib/prefix-tree" \
    -o "$BUILD_DIR/prefix-tree-py.wasm" 2>/dev/null

echo "=== Building Python recursion-of-thought ==="
bakery build \
    --inferlib "$REPO_ROOT/sdk/rust/inferlib" \
    --bindings "$REPO_ROOT/sdk/rust/inferlib/py-bindings" \
    "$REPO_ROOT/sdk/examples-py-inferlib/recursion-of-thought" \
    -o "$BUILD_DIR/recursion-of-thought-py.wasm" 2>/dev/null

echo "=== Building Python template-generation ==="
bakery build \
    --inferlib "$REPO_ROOT/sdk/rust/inferlib" \
    --bindings "$REPO_ROOT/sdk/rust/inferlib/py-bindings" \
    "$REPO_ROOT/sdk/examples-py-inferlib/template-generation" \
    -o "$BUILD_DIR/template-generation-py.wasm" 2>/dev/null

echo "=== Building Python cacheback-decoding ==="
bakery build \
    --inferlib "$REPO_ROOT/sdk/rust/inferlib" \
    --bindings "$REPO_ROOT/sdk/rust/inferlib/py-bindings" \
    "$REPO_ROOT/sdk/examples-py-inferlib/cacheback-decoding" \
    -o "$BUILD_DIR/cacheback-decoding-py.wasm" 2>/dev/null

echo "=== Building Python agent-codeact ==="
bakery build \
    --inferlib "$REPO_ROOT/sdk/rust/inferlib" \
    --bindings "$REPO_ROOT/sdk/rust/inferlib/py-bindings" \
    "$REPO_ROOT/sdk/examples-py-inferlib/agent-codeact" \
    -o "$BUILD_DIR/agent-codeact-py.wasm" 2>/dev/null

echo "=== Building Python text-completion-greedy ==="
bakery build \
    --inferlib "$REPO_ROOT/sdk/rust/inferlib" \
    --bindings "$REPO_ROOT/sdk/rust/inferlib/py-bindings" \
    "$REPO_ROOT/sdk/examples-py-inferlib/text-completion-greedy" \
    -o "$BUILD_DIR/text-completion-greedy-py.wasm" 2>/dev/null

# Pre-link static binaries using wac plug so linking time is excluded from
# the latency measurements.  Libraries are plugged in the same topological
# order that pie-client submit would use.
wac_plug() {
    # Usage: wac_plug <source.wasm> <output.wasm> <lib1.wasm> [<lib2.wasm> ...]
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
_EX="$REPO_ROOT/sdk/examples-inferlib/target/wasm32-wasip2/release"

echo "=== Pre-linking static binaries ==="
echo -n "  prefix-tree..."
wac_plug "$_EX/prefix_tree.wasm"          "$BUILD_DIR/prefix-tree-static.wasm" \
    "$_IL/inferlib_inference.wasm"
echo " done."

echo -n "  rot..."
wac_plug "$_EX/recursion_of_thought.wasm" "$BUILD_DIR/rot-static.wasm" \
    "$_IL/inferlib_inference.wasm"
echo " done."

echo -n "  template-gen..."
wac_plug "$_EX/template_generation.wasm"  "$BUILD_DIR/template-gen-static.wasm" \
    "$_IL/inferlib_inference.wasm" \
    "$_IL/inferlib_llguidance.wasm" \
    "$_IL/inferlib_schema.wasm" \
    "$_IL/inferlib_template.wasm"
echo " done."

echo -n "  cacheback..."
wac_plug "$_EX/cacheback_decoding.wasm"   "$BUILD_DIR/cacheback-static.wasm" \
    "$_IL/inferlib_inference.wasm" \
    "$_IL/inferlib_cacheback.wasm"
echo " done."

echo -n "  codeact..."
wac_plug "$_EX/agent_codeact.wasm"        "$BUILD_DIR/codeact-static.wasm" \
    "$_IL/inferlib_inference.wasm" \
    "$_IL/inferlib_js_engine.wasm"
echo " done."

echo -n "  text-completion..."
wac_plug "$_EX/text_completion_greedy.wasm" "$BUILD_DIR/text-completion-static.wasm" \
    "$_IL/inferlib_inference.wasm"
echo " done."

echo ""

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
            echo "Engine exited unexpectedly. Log:"
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
            echo -n "  Installing inferlib deps (inference)..."
            pie-client install \
                --path "$REPO_ROOT/sdk/rust/inferlib/target/wasm32-wasip2/release/inferlib_inference.wasm" \
                --manifest "$REPO_ROOT/sdk/rust/inferlib/inference/Pie.toml" > /dev/null 2>&1
            echo " done."
            ;;
        template)
            echo -n "  Installing inferlib deps (inference, llguidance, schema, template)..."
            for dep in inference llguidance schema template; do
                pie-client install \
                    --path "$REPO_ROOT/sdk/rust/inferlib/target/wasm32-wasip2/release/inferlib_${dep}.wasm" \
                    --manifest "$REPO_ROOT/sdk/rust/inferlib/$dep/Pie.toml" > /dev/null 2>&1
            done
            echo " done."
            ;;
        cacheback)
            echo -n "  Installing inferlib deps (inference, cacheback)..."
            for dep in inference cacheback; do
                pie-client install \
                    --path "$REPO_ROOT/sdk/rust/inferlib/target/wasm32-wasip2/release/inferlib_${dep}.wasm" \
                    --manifest "$REPO_ROOT/sdk/rust/inferlib/$dep/Pie.toml" > /dev/null 2>&1
            done
            echo " done."
            ;;
        codeact)
            echo -n "  Installing inferlib deps (inference, js-engine)..."
            for dep in inference js-engine; do
                local wasm_name="${dep//-/_}"
                pie-client install \
                    --path "$REPO_ROOT/sdk/rust/inferlib/target/wasm32-wasip2/release/inferlib_${wasm_name}.wasm" \
                    --manifest "$REPO_ROOT/sdk/rust/inferlib/$dep/Pie.toml" > /dev/null 2>&1
            done
            echo " done."
            ;;
        *)
            echo "Unknown dep_set: $dep_set" >&2
            exit 1
            ;;
    esac
}

# Submit the application and echo elapsed milliseconds to stdout.
timed_submit() {
    local wasm_path="$1"
    local manifest_path="$2"
    local extra_args="${3:-}"
    local t1 t2
    t1=$(date +%s%3N)
    # shellcheck disable=SC2086
    pie-client submit \
        --path "$wasm_path" \
        --manifest "$manifest_path" \
        $extra_args > /dev/null 2>&1
    t2=$(date +%s%3N)
    echo $((t2 - t1))
}

# ---------------------------------------------------------------------------
# Phase 2: Run each case  —  1 warm-up + REPEATS measured runs per engine start
# ---------------------------------------------------------------------------

declare -A all_times

for entry in "${CASES[@]}"; do
    IFS='|' read -r name wasm_path manifest_path dep_set extra_args <<< "$entry"

    echo "============================================================"
    echo "  $name  ($REPEATS runs + 1 warm-up)"
    echo "============================================================"

    echo -n "  Starting engine..."
    start_engine
    echo " ready."

    install_deps "$dep_set"

    echo -n "  Warm-up run..."
    timed_submit "$wasm_path" "$manifest_path" "$extra_args" > /dev/null
    echo " done."

    case_times=""
    for i in $(seq 1 $REPEATS); do
        echo -n "  Run $i/$REPEATS ... "
        ms=$(timed_submit "$wasm_path" "$manifest_path" "$extra_args")
        echo "${ms} ms"
        case_times="$case_times $ms"
    done

    all_times["$name"]="$case_times"

    python3 -c "
times = [float(x) for x in '''$case_times'''.split()]
avg = sum(times) / len(times)
mn  = min(times)
mx  = max(times)
print()
print(f'  Average: {avg:.0f} ms   min: {mn:.0f} ms   max: {mx:.0f} ms')
"
    echo ""
    stop_engine
done

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

echo "============================================================"
echo "  Summary ($REPEATS measured runs each, 1 warm-up excluded)"
echo "============================================================"
for entry in "${CASES[@]}"; do
    IFS='|' read -r name _ _ _ _ <<< "$entry"
    python3 -c "
times = [float(x) for x in '''${all_times[$name]}'''.split()]
avg = sum(times) / len(times)
mn  = min(times)
mx  = max(times)
individual = '  '.join(f'{t:.0f}' for t in times)
print(f'  {\"$name\":<36}  avg={avg:>6.0f} ms   min={mn:>6.0f} ms   max={mx:>6.0f} ms   [{individual}] ms')
"
done
