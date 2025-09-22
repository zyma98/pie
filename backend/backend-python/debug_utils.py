"""Debug utilities for Metal and Python backends."""

import os

# Import debug framework checkpoint decorator
try:
    from debug_framework.decorators.checkpoint_decorator import checkpoint_validation
except ImportError:
    def checkpoint_validation(*args, **kwargs):
        """Fallback no-op decorator when debug framework is not available."""
        def decorator(func):
            return func
        return decorator


def is_tensor_debug_enabled() -> bool:
    """Check if MetalTensorDebug logging is enabled via environment variable."""
    return os.environ.get("METAL_DEBUG_TENSOR", "0").lower() in {"1", "true", "yes", "on"}


def is_capture_debug_enabled() -> bool:
    """Check if MetalCaptureDebug logging is enabled via environment variable."""
    return os.environ.get("METAL_DEBUG_CAPTURE", "0").lower() in {"1", "true", "yes", "on"}