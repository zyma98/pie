#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
ENGINE_LOG=$(mktemp)
ENGINE_PID=""
REPEATS=10

cleanup() {
    if [ -n "$ENGINE_PID" ] && kill -0 "$ENGINE_PID" 2>/dev/null; then
        echo ""
        echo "=== Stopping engine (PID $ENGINE_PID) ==="
        kill -TERM "$ENGINE_PID"
        wait "$ENGINE_PID" 2>/dev/null || true
        echo "Engine stopped."
    fi
    rm -f "$ENGINE_LOG"
}
trap cleanup EXIT

source "$REPO_ROOT/pie/.venv/bin/activate"

# ---------------------------------------------------------------------------
# Test cases: name | wasm path (relative to REPO_ROOT) | manifest path | num deps | extra submit args
# Ordered by number of inferlib dependencies (0 -> 4)
# ---------------------------------------------------------------------------
CASES=(
    "text-completion|std/text-completion/target/wasm32-wasip2/release/text_completion.wasm|std/text-completion/Pie.toml|0|-- -p Hello"
    "text-completion-inferlib|sdk/examples-inferlib/target/wasm32-wasip2/release/text_completion_inferlib.wasm|sdk/examples-inferlib/text-completion-inferlib/Pie.toml|1|-- -p Hello"
    "constrained-decoding|sdk/examples-inferlib/target/wasm32-wasip2/release/constrained_decoding.wasm|sdk/examples-inferlib/constrained-decoding/Pie.toml|2|"
    "json-schema-validation|sdk/examples-inferlib/target/wasm32-wasip2/release/json_schema_validation.wasm|sdk/examples-inferlib/json-schema-validation/Pie.toml|3|"
    "template-generation|sdk/examples-inferlib/target/wasm32-wasip2/release/template_generation.wasm|sdk/examples-inferlib/template-generation/Pie.toml|4|"
)

# ---------------------------------------------------------------------------
# Phase 1: Build
# ---------------------------------------------------------------------------

echo "=== Building runtime with case_study_instantiation_time feature ==="
cd "$REPO_ROOT/pie"
if ! maturin develop --release --features case_study_instantiation_time > /dev/null 2>&1; then
    echo "maturin build failed! Re-running with output:"
    maturin develop --release --features case_study_instantiation_time
    exit 1
fi

echo "=== Building inferlib dependencies ==="
cd "$REPO_ROOT/sdk/rust/inferlib"
cargo build -rq --target wasm32-wasip2 2>/dev/null

echo "=== Building std/text-completion ==="
cd "$REPO_ROOT/std/text-completion"
cargo build -rq --target wasm32-wasip2 2>/dev/null

echo "=== Building sdk/examples-inferlib cases ==="
cd "$REPO_ROOT/sdk/examples-inferlib"
cargo build -rq --target wasm32-wasip2 \
    -p text-completion-inferlib \
    -p constrained-decoding \
    -p json-schema-validation \
    -p template-generation \
    2>/dev/null

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

install_inferlib_deps() {
    cd "$REPO_ROOT/sdk/rust/inferlib"
    pie-client install \
        --path target/wasm32-wasip2/release/inferlib_inference.wasm \
        --manifest inference/Pie.toml > /dev/null 2>&1
    pie-client install \
        --path target/wasm32-wasip2/release/inferlib_llguidance.wasm \
        --manifest llguidance/Pie.toml > /dev/null 2>&1
    pie-client install \
        --path target/wasm32-wasip2/release/inferlib_schema.wasm \
        --manifest schema/Pie.toml > /dev/null 2>&1
    pie-client install \
        --path target/wasm32-wasip2/release/inferlib_template.wasm \
        --manifest template/Pie.toml > /dev/null 2>&1
}

submit_app() {
    local wasm_path="$1"
    local manifest_path="$2"
    local extra_args="${3:-}"
    # shellcheck disable=SC2086
    pie-client submit \
        --path "$wasm_path" \
        --manifest "$manifest_path" \
        $extra_args > /dev/null 2>&1
}

extract_us() {
    local label="$1"
    grep "\[case-study\].*$label" "$ENGINE_LOG" | \
        sed "s/.*$label: *\([0-9.]*\) us/\1/"
}

# ---------------------------------------------------------------------------
# Phase 2: Run each case REPEATS times
# ---------------------------------------------------------------------------

# Associative arrays to accumulate values per case
declare -A all_store_linker
declare -A all_dep
declare -A all_app
declare -A all_total

for entry in "${CASES[@]}"; do
    IFS='|' read -r name wasm_rel manifest_rel num_deps extra_args <<< "$entry"
    wasm_path="$REPO_ROOT/$wasm_rel"
    manifest_path="$REPO_ROOT/$manifest_rel"

    echo ""
    echo "============================================================"
    echo "  ${name} (${num_deps} inferlib deps, ${REPEATS} runs)"
    echo "============================================================"

    case_sl=""
    case_dep=""
    case_app=""
    case_tot=""

    for i in $(seq 1 $REPEATS); do
        echo -n "  Run $i/$REPEATS: starting engine..."
        start_engine
        echo -n " installing..."
        install_inferlib_deps
        echo -n " submitting..."
        submit_app "$wasm_path" "$manifest_path" "$extra_args"

        sl=$(extract_us "Store + Linker creation")
        dep=$(extract_us "Dependency instantiation")
        app=$(extract_us "App component instantiation")
        tot=$(extract_us "Total")

        echo " store+linker=${sl} us, deps=${dep} us, app=${app} us, total=${tot} us"

        case_sl="$case_sl $sl"
        case_dep="$case_dep $dep"
        case_app="$case_app $app"
        case_tot="$case_tot $tot"

        stop_engine
    done

    all_store_linker["$name"]="$case_sl"
    all_dep["$name"]="$case_dep"
    all_app["$name"]="$case_app"
    all_total["$name"]="$case_tot"

    python3 -c "
sl = [float(x) for x in '''$case_sl'''.split()]
dep = [float(x) for x in '''$case_dep'''.split()]
app = [float(x) for x in '''$case_app'''.split()]
tot = [float(x) for x in '''$case_tot'''.split()]
print()
print('  --- Average ---')
print(f'  Store + Linker creation:       {sum(sl)/len(sl):.1f} us')
print(f'  Dependency instantiation:      {sum(dep)/len(dep):.1f} us')
print(f'  App component instantiation:   {sum(app)/len(app):.1f} us')
print(f'  Total:                         {sum(tot)/len(tot):.1f} us')
"
done

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

echo ""
echo "============================================================"
echo "  Overall Summary ($REPEATS runs each)"
echo "============================================================"

for entry in "${CASES[@]}"; do
    IFS='|' read -r name _ _ num_deps _ <<< "$entry"
    python3 -c "
tot = [float(x) for x in '''${all_total[$name]}'''.split()]
dep = [float(x) for x in '''${all_dep[$name]}'''.split()]
app = [float(x) for x in '''${all_app[$name]}'''.split()]
sl  = [float(x) for x in '''${all_store_linker[$name]}'''.split()]
print(f'  ${name} (${num_deps} deps):')
print(f'    store+linker={sum(sl)/len(sl):.1f}  deps={sum(dep)/len(dep):.1f}  app={sum(app)/len(app):.1f}  total={sum(tot)/len(tot):.1f} us')
"
done
