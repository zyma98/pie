"""
inference_bindings - Thin wrapper over inferlib WIT-generated bindings.

Minimal subset needed for microbenchmarking: argument retrieval and return.
"""

from wit_world.imports import runtime as _runtime

get_arguments = _runtime.get_arguments
set_return = _runtime.set_return
