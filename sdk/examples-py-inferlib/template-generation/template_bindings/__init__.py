"""
template_bindings - Thin Python wrapper over inferlib-template WIT bindings.

Pure re-exports of WIT resource types.
Logically equivalent to Rust's inferlib-template-bindings crate.
"""

from wit_world.imports import template_rendering as _tr

TemplateRenderer = _tr.TemplateRenderer
