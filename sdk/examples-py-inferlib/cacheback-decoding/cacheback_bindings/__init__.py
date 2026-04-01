"""
cacheback_bindings - Thin Python wrapper over inferlib-cacheback WIT bindings.

Pure re-exports of WIT resource types.
Logically equivalent to Rust's inferlib-cacheback-bindings crate.
"""

from wit_world.imports import cacheback as _cb

CacheTable = _cb.CacheTable
