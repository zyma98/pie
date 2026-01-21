#!/bin/bash
set -e

# Run from the workspace root so paths are correct
cd "$(dirname "$0")"

# Default paths
LOGGING="target/wasm32-wasip2/release/provider_logging.wasm"
CALCULATOR="target/wasm32-wasip2/release/provider_calculator.wasm"
APP="target/wasm32-wasip2/release/app_consumer.wasm"

if [ "$1" = "--demo" ] || [ "$1" = "-d" ]; then
    # Run the standard demo by piping commands to the host
    echo "Running ComponentHub demo..."
    echo
    {
        echo "load $LOGGING"
        echo "load $CALCULATOR"
        echo "run $APP"
        echo "quit"
    } | cargo run --package host --release
elif [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    echo "Usage: $0 [OPTIONS]"
    echo
    echo "Options:"
    echo "  --demo, -d     Run the standard demo (load libraries and run app)"
    echo "  --help, -h     Show this help message"
    echo "  (no args)      Start interactive mode"
    echo
    echo "In interactive mode, use these commands:"
    echo "  load <path>    Load a library WASM"
    echo "  run <path>     Run an application WASM"
    echo "  status         Show loaded libraries"
    echo "  help           Show available commands"
    echo "  quit/exit      Exit the host"
else
    # Start interactive mode
    echo "Starting ComponentHub in interactive mode..."
    echo
    cargo run --package host --release
fi
