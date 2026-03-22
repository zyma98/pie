"""
schema_bindings - Thin Python wrapper over inferlib-schema WIT bindings.

Pure re-exports of WIT resource types.
Logically equivalent to Rust's inferlib-schema-bindings crate.
"""

from wit_world.imports import json_schema as _js

SchemaValidator = _js.SchemaValidator
