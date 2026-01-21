# WebAssembly Component Model Dynamic Linking Demo

## Overview

This repository demonstrates a **monotonic dynamic-linking host** for the WebAssembly Component Model using wasmtime (Rust). It shows how to dynamically link WebAssembly components at runtime, with full support for:

- **WIT resources** with ownership semantics (own/borrow) and proper destructor handling
- **Library dependency chains** - libraries can depend on other libraries
- **All function types** - constructors, instance methods, static functions, and standalone functions
- **Interactive command interface** - load libraries, run apps, and manage state via stdin
- **Persistent library state** - libraries remain loaded and maintain state across app runs

## Architecture

The system demonstrates a dependency chain:

```
┌─────────────────┐     ┌─────────────────────┐     ┌─────────────────┐
│  app_consumer   │────▶│ provider_calculator │────▶│ provider_logging│
│  (imports calc) │     │ (imports logging,   │     │ (exports        │
│                 │     │  exports calculator)│     │  logging)       │
└─────────────────┘     └─────────────────────┘     └─────────────────┘
```

### 1. Host (`host/`) - ComponentHub

An interactive Rust binary that acts as a dynamic linker for WebAssembly components. It:
- Accepts commands via stdin to load libraries and run applications
- Loads components one-by-one in dependency order
- Scans provider components to discover their exports (interfaces, resources, functions)
- Registers forwarding implementations that proxy calls from consumers to providers
- Handles resource lifecycle with proper destructor chaining
- Maintains persistent state across multiple app runs

### 2. Provider Components

#### `provider_logging/`
Exports a logging interface with:
- **Resource**: `logger` with constructor, methods, and destructor
- **Static function**: `get-logger-count` (on resource)
- **Standalone functions**: `get-default-level`, `level-to-string`

#### `provider_calculator/`
Imports logging and exports a calculator interface with:
- **Resource**: `calc` with constructor, methods, and destructor
- **Static functions**: `version`, `total-operations` (on resource)
- **Standalone functions**: `pi`, `quick-add`
- Internally uses the logging interface for operation logging

### 3. App Component (`app_consumer/`)
Imports both logging and calculator interfaces to demonstrate:
- Using standalone functions (no resource needed)
- Using static functions (on resource types)
- Creating and using resource instances
- Proper resource cleanup (triggering destructor chain)

## WIT Interfaces

```wit
package demo:logging@0.1.0;

interface logging {
    enum level { debug, info, warn, error }

    // Standalone functions
    get-default-level: func() -> level;
    level-to-string: func(lvl: level) -> string;

    resource logger {
        constructor(max-level: level);
        get-logger-count: static func() -> u32;  // Static function
        get-max-level: func() -> level;           // Instance method
        set-max-level: func(level: level);
        log: func(level: level, msg: string);
    }
}

interface calculator {
    // Standalone functions
    pi: func() -> f64;
    quick-add: func(a: f64, b: f64) -> f64;

    resource calc {
        constructor();
        version: static func() -> string;           // Static function
        total-operations: static func() -> u64;
        add: func(a: f64, b: f64) -> f64;           // Instance methods
        subtract: func(a: f64, b: f64) -> f64;
        multiply: func(a: f64, b: f64) -> f64;
        divide: func(a: f64, b: f64) -> f64;
    }
}
```

## Host Commands

The host accepts commands via stdin:

| Command | Description |
|---------|-------------|
| `load <path>` | Load a library WASM and register its exports |
| `run <path>` | Run an application WASM (must export `run` function) |
| `status` | Show loaded libraries |
| `purge` | Remove all loaded libraries and reset state |
| `help` | Show available commands |
| `quit` / `exit` | Exit the host |

## Key Technical Features

### 1. Function Type Handling

The host categorizes and forwards different function types:
- **Constructors** (`[constructor]resource`): Create resources, map handles
- **Instance methods** (`[method]resource.name`): Translate resource handles, forward calls
- **Static functions** (`[static]resource.name`): Forward directly
- **Standalone functions** (plain names): Forward directly

### 2. Resource Forwarding

Resources use handle-based representation. The host maintains mappings between:
- Host-side resource handles (seen by consumers)
- Provider's `ResourceAny` handles (actual implementation)

### 2.1 Resource Type Transformation (Incoming/Outgoing)

The host must translate resource handles when values cross component boundaries. The
transform is **type-directed** and must distinguish *owned* resources (defined by the
current interface) from *imported* resources (defined elsewhere):

- **Incoming (consumer → provider)**: walk the argument values by their WIT types.
  - If a value is a resource of a type *owned by this provider*, unwrap the host
    handle to the provider's `ResourceAny`.
  - If the resource type is *imported*, keep the host handle intact to preserve
    cross-provider compatibility.
- **Outgoing (provider → consumer)**: walk return values by their WIT types.
  - If a value is a resource of a type *owned by this provider*, wrap it into a
    host resource handle and record the `rep → ResourceAny` mapping.
  - Imported resources pass through unchanged.

This recursive transform applies to nested containers (e.g., `option`, `result`,
`list`, `record`, `tuple`, `variant`) so that resource handles are translated
correctly no matter where they appear.

#### Why ownership matters (cross-provider example)

If provider **X** defines resource `logger` and provider **Y** accepts `logger` as an
argument, then Y expects **host** handles for `logger` (because it *imports* the type).
If the host unwraps `logger` when calling Y, Y will receive X's raw `ResourceAny` and
subsequent method calls will fail with mismatched resource types. The ownership-aware
transform avoids this by only unwrapping resource types that the current interface
actually defines.

### 3. Destructor Chaining

When a consumer drops a resource:
1. Host's destructor callback is invoked
2. Host looks up the provider's `ResourceAny`
3. Host calls `resource_drop_async()` on the provider
4. Provider's Rust `Drop` implementation runs
5. Nested resources are recursively dropped (e.g., calc drops its logger)

### 4. Persistent State

Libraries remain instantiated between `run` commands:
- Static variables in libraries persist
- Counters and other state accumulate
- Use `purge` to reset everything

### 5. Performance Optimization

Function handles (`Func`) are resolved once during registration and captured in closures, avoiding lookups at call time.

## Prerequisites

- Rust (with cargo)
- wasm32-wasip2 target: `rustup target add wasm32-wasip2`

## Building & Running

```bash
# Build everything
./build.sh

# Run the standard demo (non-interactive)
./run.sh --demo

# Start interactive mode
./run.sh

# Or directly with cargo
cargo run --package host --release
```

### Interactive Session Example

```
> load target/wasm32-wasip2/release/provider_logging.wasm
=== Library loaded successfully! ===

> load target/wasm32-wasip2/release/provider_calculator.wasm
=== Library loaded successfully! ===

> status
Loaded libraries (2):
  1. target/wasm32-wasip2/release/provider_logging.wasm
  2. target/wasm32-wasip2/release/provider_calculator.wasm

> run target/wasm32-wasip2/release/app_consumer.wasm
[APP] run() starting...
...
=== App execution completed successfully! ===

> run target/wasm32-wasip2/release/app_consumer.wasm
[APP] run() starting...
... (state persists from previous run)
=== App execution completed successfully! ===

> purge
Purged 2 libraries. All state has been reset.

> quit
Goodbye!
```

## Output Example

The demo shows interleaved output from multiple layers:
- `[HOST]` - The forwarding/linking layer
- `[PROVIDER]` - The logging implementation
- `[CALCULATOR]` - The calculator implementation
- `[APP]` - The consumer application

```
[APP] === Testing calculator standalone functions ===
[HOST] Static function pi called with 0 args
[CALCULATOR] pi() -> 3.141592653589793
[APP] Value of pi: 3.141592653589793

[APP] === Testing calculator static functions ===
[HOST] Static function [static]calc.version called with 0 args
[CALCULATOR] Calc::version() -> "1.0.0"
[APP] Calculator version: "1.0.0"

[APP] Creating calculator
[HOST] Constructor [constructor]calc called with 0 args
[HOST] Constructor [constructor]logger called with 1 args
[PROVIDER] Logger::new(id=1, max_level=DEBUG) - constructor called
[HOST] Method [method]logger.log called with 3 args
[PROVIDER] Logger::log(id=1, level=INFO) -> LOGGED: [CALCULATOR] Calc::new(id=1)

[APP] About to drop calculator...
[HOST] Resource destructor called: demo:logging/calculator@0.1.0::calc
[HOST] Method [method]logger.log called with 3 args
[PROVIDER] Logger::log(id=1, level=INFO) -> LOGGED: [CALCULATOR] Calc::drop(id=1)
[HOST] Resource destructor called: demo:logging/logging@0.1.0::logger
[PROVIDER] Logger::drop(id=1) - destructor called!
```

## Project Structure

```
wasm-dyn-link2/
├── Cargo.toml              # Workspace definition
├── wit/world.wit           # Shared WIT interface definitions
├── host/                   # ComponentHub host binary
│   ├── Cargo.toml
│   └── src/main.rs
├── provider_logging/       # Logging provider (exports logging)
│   ├── Cargo.toml
│   └── src/lib.rs
├── provider_calculator/    # Calculator provider (imports logging, exports calculator)
│   ├── Cargo.toml
│   └── src/lib.rs
├── app_consumer/           # App (imports logging + calculator)
│   ├── Cargo.toml
│   └── src/lib.rs
├── build.sh               # Build script
├── run.sh                 # Run script (supports --demo and interactive modes)
└── README.md
```

## Limitations

### No Selective Unload

Individual libraries cannot be unloaded because:
- The wasmtime `Linker` API doesn't support removing registered definitions
- Captured `Func` handles would become invalid
- Active resource handles would dangle
- Dependency tracking would be complex

**Workaround**: Use the `purge` command to reset everything and reload libraries.

## License

MIT
