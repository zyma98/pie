"""
Unified profiling utilities for PyTorch models.

This package combines timing profiling, memory tracking, and tensor flow analysis.
Exports APIs from both the legacy profiler and the unified tracker.
"""

from contextlib import contextmanager

# Import from unified tracker (main profiler)
from .tracker import (
    get_memory_summary,
    get_memory_tracker,
    initialize_memory_tracker,
    memory_checkpoint,
    save_memory_profile,
    stop_memory_tracker,
)

# Import from legacy profiler (for backward compatibility)
from .legacy_profiler import (
    set_profiling_enabled,
    profile_with_tensors,
)


@contextmanager
def start_profile(name: str):
    """
    Profile an operation with timing and optional tensor tracking (unified profiler).

    This is a convenience wrapper that delegates to the global memory tracker's
    start_profile method. Compatible with the old profiler.start_profile() API.

    Usage:
        from profiler import start_profile

        with start_profile("operation_name"):
            # code to profile

    Args:
        name: Name of the operation (e.g., "forward_pass", "attn_metal_kernel")
    """
    tracker = get_memory_tracker()
    with tracker.start_profile(name):
        yield


__all__ = [
    # Unified tracker API
    "initialize_memory_tracker",
    "get_memory_tracker",
    "memory_checkpoint",
    "save_memory_profile",
    "get_memory_summary",
    "stop_memory_tracker",
    "start_profile",

    # Legacy profiler API (backward compatibility)
    "set_profiling_enabled",
    "profile_with_tensors",
]
