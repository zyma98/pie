"""
cacheback_py_bindings - Thin Python wrapper over inferlib-cacheback-py WIT bindings.

Pure re-exports of WIT resource types.
Logically equivalent to Rust's inferlib-cacheback-py-bindings crate.
"""

from wit_world.imports import cacheback as _cb

CacheTable = _cb.CacheTable
DraftResult = _cb.DraftResult
