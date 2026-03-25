"""
echo_callee_bindings - Thin wrapper over the echo-callee WIT bindings.

Re-exports the echo function from the microbench:echo-callee/echo interface.
"""

from wit_world.imports.echo import echo

__all__ = ["echo"]
