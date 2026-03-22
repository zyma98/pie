"""
llguidance_bindings - Thin Python wrapper over inferlib-llguidance WIT bindings.

Pure re-exports of WIT resource types.
Logically equivalent to Rust's inferlib-llguidance-bindings crate.
"""

from wit_world.imports import constrained_sampling as _cs

GrammarMatcher = _cs.GrammarMatcher
TokenMask = _cs.TokenMask
