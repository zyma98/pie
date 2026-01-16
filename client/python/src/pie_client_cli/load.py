"""Load command implementation for the Pie CLI.

This module implements the `pie-cli load` subcommand for loading
library components into a Pie engine.
"""

from pathlib import Path
from typing import Optional

import typer

from . import engine


def handle_load_command(
    path: Path,
    name: Optional[str] = None,
    dependencies: Optional[list[str]] = None,
    config: Optional[Path] = None,
    host: Optional[str] = None,
    port: Optional[int] = None,
    username: Optional[str] = None,
    private_key_path: Optional[Path] = None,
) -> None:
    """Handle the `pie-cli load` command.

    1. Creates a client configuration from config file and command-line arguments
    2. Connects to the Pie engine server
    3. Reads the library WASM file
    4. Uploads the library with the specified name and dependencies
    """
    if not path.exists():
        raise FileNotFoundError(f"Library file not found: {path}")

    # Use the file stem as the library name if not specified
    library_name = name if name else path.stem

    dependencies = dependencies or []

    client_config = engine.ClientConfig.create(
        config_path=config,
        host=host,
        port=port,
        username=username,
        private_key_path=private_key_path,
    )

    client = engine.connect_and_authenticate(client_config)

    try:
        # Read the library bytes
        library_bytes = path.read_bytes()

        typer.echo(f"Loading library '{library_name}' from {path}")
        if dependencies:
            typer.echo(f"  Dependencies: {', '.join(dependencies)}")

        # Upload the library
        engine.upload_library(client, library_name, library_bytes, dependencies)

        typer.echo(f"✅ Library '{library_name}' loaded successfully.")

    finally:
        engine.close_client(client)
