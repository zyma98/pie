"""Load command implementation for the Pie CLI.

This module implements the `pie-cli load` subcommand for loading
library components into a Pie engine.
"""

from pathlib import Path
from typing import Optional

import typer

from . import engine


def handle_load_command(
    library: Optional[str] = None,
    path: Optional[Path] = None,
    name: Optional[str] = None,
    dependencies: Optional[list[str]] = None,
    config: Optional[Path] = None,
    host: Optional[str] = None,
    port: Optional[int] = None,
    username: Optional[str] = None,
    private_key_path: Optional[Path] = None,
) -> None:
    """Handle the `pie-cli load` command.

    You can specify a library either by registry name or by path (mutually exclusive):

    - By registry: pie-client load std/my-library@0.1.0
    - By path: pie-client load --path ./my_library.wasm

    Steps:
    1. Creates a client configuration from config file and command-line arguments
    2. Connects to the Pie engine server
    3. If using path, reads the library WASM file and uploads it
    4. If using registry, downloads and loads the library from the registry
    """
    # Validate at least one of library or path is provided
    if library is None and path is None:
        typer.echo("Error: Specify a library name or --path", err=True)
        raise typer.Exit(1)

    # Validate mutual exclusivity
    if library is not None and path is not None:
        typer.echo("Error: Cannot specify both library name and --path", err=True)
        raise typer.Exit(1)

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
        if path is not None:
            # Load from local file path
            if not path.exists():
                raise FileNotFoundError(f"Library file not found: {path}")

            # Use the file stem as the library name if not specified
            library_name = name if name else path.stem

            # Read the library bytes
            library_bytes = path.read_bytes()

            typer.echo(f"Loading library '{library_name}' from {path}")
            if dependencies:
                typer.echo(f"  Dependencies: {', '.join(dependencies)}")

            # Upload the library
            engine.upload_library(client, library_name, library_bytes, dependencies)

            typer.echo(f"✅ Library '{library_name}' loaded successfully.")
        else:
            # Load from registry
            if name is not None:
                typer.echo(
                    "Warning: --name option is ignored when loading from registry",
                    err=True,
                )

            typer.echo(f"Loading library from registry: {library}")
            if dependencies:
                typer.echo(f"  Dependencies: {', '.join(dependencies)}")

            engine.load_library_from_registry(client, library, dependencies)

            typer.echo(f"✅ Library '{library}' loaded successfully from registry.")

    finally:
        engine.close_client(client)
