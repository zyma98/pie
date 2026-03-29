#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
MICROBENCH_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
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
# Phase 1: Build
# ---------------------------------------------------------------------------

echo "=== Building runtime with microbench_snapshot feature ==="
cd "$REPO_ROOT/pie"
if ! maturin develop --release --features microbench_snapshot > /dev/null 2>&1; then
    echo "maturin build failed! Re-running with output:"
    maturin develop --release --features microbench_snapshot
    exit 1
fi

echo "=== Building inferlib dependencies ==="
cd "$REPO_ROOT/sdk/rust/inferlib"
cargo build -rq --target wasm32-wasip2 2>/dev/null

echo "=== Building microbench Rust crates ==="
cd "$MICROBENCH_DIR"
cargo build -rq --target wasm32-wasip2 -p snapshot-echo-caller 2>/dev/null

echo "=== Building snapshot-echo-callee (Python, shared modules) ==="
PY_CALLEE_OUT="$MICROBENCH_DIR/target/wasm32-wasip2/release"
bakery build --lib --world echo-provider \
    "$SCRIPT_DIR/echo-callee" \
    -o "$PY_CALLEE_OUT/snapshot_echo_callee.wasm"

start_engine() {
    local extra_flags="${1:-}"
    : > "$ENGINE_LOG"
    cd "$REPO_ROOT/pie"
    # shellcheck disable=SC2086
    pie serve $extra_flags > "$ENGINE_LOG" 2>&1 &
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

install_components() {
    cd "$REPO_ROOT/sdk/rust/inferlib"
    pie-client install \
        --path target/wasm32-wasip2/release/inferlib_inference.wasm \
        --manifest inference/Pie.toml 2>/dev/null

    cd "$MICROBENCH_DIR"
    pie-client install \
        --path target/wasm32-wasip2/release/snapshot_echo_callee.wasm \
        --manifest snapshot-latency/echo-callee/Pie.toml 2>/dev/null
}

run_caller() {
    cd "$MICROBENCH_DIR"
    pie-client submit \
        --path target/wasm32-wasip2/release/snapshot_echo_caller.wasm \
        --manifest snapshot-latency/echo-caller/Pie.toml 2>/dev/null
}

# Extract "Per-call latency: NNN.N ns" from caller output, convert to us
extract_call_latency_us() {
    local output="$1"
    echo "$output" | grep "Per-call latency" | \
        sed 's/.*Per-call latency: *\([0-9.]*\) ns/\1/' | \
        python3 -c "import sys; print(float(sys.stdin.read().strip()) / 1000.0)"
}

# Extract "[microbench] Snapshot creation latency: NNN.N us" from engine log
extract_snapshot_latency_us() {
    grep "\[microbench\] Snapshot creation latency" "$ENGINE_LOG" | \
        sed 's/.*Snapshot creation latency: *\([0-9.]*\) us/\1/'
}

# ---------------------------------------------------------------------------
# Phase 2: WITH snapshot optimization (10 runs)
# ---------------------------------------------------------------------------

echo ""
echo "============================================================"
echo "  Phase 2: WITH snapshot optimization ($REPEATS runs)"
echo "============================================================"

snapshot_latencies=""
with_snapshot_latencies=""

for i in $(seq 1 $REPEATS); do
    echo -n "  Run $i/$REPEATS: starting engine..."
    start_engine ""
    echo -n " installing..."
    install_components
    echo -n " running..."
    caller_output=$(run_caller)

    call_us=$(extract_call_latency_us "$caller_output")
    snap_us=$(extract_snapshot_latency_us)

    echo " call=${call_us} us, snapshot=${snap_us} us"

    with_snapshot_latencies="$with_snapshot_latencies $call_us"
    snapshot_latencies="$snapshot_latencies $snap_us"

    stop_engine
done

echo ""
echo "=== Results: WITH snapshot ==="
python3 -c "
vals = [float(x) for x in '''$with_snapshot_latencies'''.split()]
print(f'  First-call latencies (us): {[\"{:.1f}\".format(v) for v in vals]}')
print(f'  Average first-call latency: {sum(vals)/len(vals):.1f} us')
"
python3 -c "
vals = [float(x) for x in '''$snapshot_latencies'''.split()]
print(f'  Snapshot creation latencies (us): {[\"{:.1f}\".format(v) for v in vals]}')
print(f'  Average snapshot creation latency: {sum(vals)/len(vals):.1f} us')
"

# ---------------------------------------------------------------------------
# Phase 3: WITHOUT snapshot optimization (10 runs)
# ---------------------------------------------------------------------------

echo ""
echo "============================================================"
echo "  Phase 3: WITHOUT snapshot optimization ($REPEATS runs)"
echo "============================================================"

without_snapshot_latencies=""

for i in $(seq 1 $REPEATS); do
    echo -n "  Run $i/$REPEATS: starting engine..."
    start_engine "--no-snapshot"
    echo -n " installing..."
    install_components
    echo -n " running..."
    caller_output=$(run_caller)

    call_us=$(extract_call_latency_us "$caller_output")

    echo " call=${call_us} us"

    without_snapshot_latencies="$without_snapshot_latencies $call_us"

    stop_engine
done

echo ""
echo "=== Results: WITHOUT snapshot ==="
python3 -c "
vals = [float(x) for x in '''$without_snapshot_latencies'''.split()]
print(f'  First-call latencies (us): {[\"{:.1f}\".format(v) for v in vals]}')
print(f'  Average first-call latency: {sum(vals)/len(vals):.1f} us')
"

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

echo ""
echo "============================================================"
echo "  Summary"
echo "============================================================"
python3 -c "
with_vals = [float(x) for x in '''$with_snapshot_latencies'''.split()]
without_vals = [float(x) for x in '''$without_snapshot_latencies'''.split()]
snap_vals = [float(x) for x in '''$snapshot_latencies'''.split()]
avg_with = sum(with_vals) / len(with_vals)
avg_without = sum(without_vals) / len(without_vals)
avg_snap = sum(snap_vals) / len(snap_vals)
print(f'  Avg first-call latency WITH snapshot:    {avg_with:.1f} us')
print(f'  Avg first-call latency WITHOUT snapshot: {avg_without:.1f} us')
print(f'  Avg snapshot creation latency:           {avg_snap:.1f} us')
print(f'  Speedup from snapshot:                   {avg_without/avg_with:.1f}x')
"
