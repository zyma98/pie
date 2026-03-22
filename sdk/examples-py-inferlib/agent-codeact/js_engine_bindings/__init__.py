"""
js_engine_bindings - Thin Python wrapper over inferlib-js-engine WIT bindings.

Pure re-exports of WIT functions.
Logically equivalent to Rust's inferlib-js-engine-bindings crate.
"""

from wit_world.imports import js_engine as _js

execute = _js.execute
