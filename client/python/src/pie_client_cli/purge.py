"""Purge command implementation for the Pie CLI.

This module implements the `pie-cli purge` subcommand for removing
all loaded libraries from a Pie engine.
"""

from pathlib import Path
from typing import Optional

import typer

from . import engine


def handle_purge_command(
    config: Optional[Path] = None,
    host: Optional[str] = None,
    port: Optional[int] = None,
    username: Optional[str] = None,
    private_key_path: Optional[Path] = None,
) -> None:
    """Handle the `pie-cli purge` command.

    Removes all loaded libraries from the Pie engine.
    This operation is only allowed when no instances are running.

    Steps:
    1. Creates a client configuration from config file and command-line arguments
    2. Connects to the Pie engine server
    3. Sends the purge command to remove all loaded libraries
    """
    client_config = engine.ClientConfig.create(
        config_path=config,
        host=host,
        port=port,
        username=username,
        private_key_path=private_key_path,
    )

    client = engine.connect_and_authenticate(client_config)

    try:
        typer.echo("Purging all loaded libraries...")

        count = engine.purge_libraries(client)

        if count == 0:
            typer.echo("No libraries were loaded.")
        elif count == 1:
            typer.echo("✅ Purged 1 library.")
        else:
            typer.echo(f"✅ Purged {count} libraries.")

    finally:
        engine.close_client(client)
