"""Tensor analysis utilities for profiling."""

from __future__ import annotations

import torch


def classify_tensor_purpose(tensor: torch.Tensor, shape: tuple) -> str:
    """Classify what this tensor is used for."""
    # KV cache detection: large 5D tensors with specific shape pattern
    if (
        len(shape) == 5 and shape[0] > 4096
    ):  # [max_cache_len, batch, num_kv_heads, num_heads_per_group, head_dim]
        return "kv_cache"

    # Weight detection: requires_grad or very large
    if tensor.requires_grad:
        if tensor.numel() > 1000000:  # > 1M parameters
            return "weight"
        return "gradient"

    # Embedding detection: 2D with large vocabulary size
    if len(shape) == 2 and shape[0] > 10000:
        return "embedding"

    # Intermediate buffer: modest size, temporary
    if len(shape) >= 2:
        return "buffer"

    return "unknown"


def is_persistent_tensor(
    tensor_id: int, size_mb: float, tensor_last_seen: dict[int, str]
) -> bool:
    """Determine if tensor persists across operations."""
    # If we've seen this tensor before, it's persistent
    if tensor_id in tensor_last_seen:
        return True

    # Large tensors (>100MB) are likely persistent (weights, cache)
    if size_mb > 100:
        return True

    return False


def is_reusable_tensor(tensor: torch.Tensor, purpose: str) -> bool:
    """Determine if tensor memory can be reused."""
    # Weights and cache are not reusable
    if purpose in ["weight", "kv_cache", "embedding"]:
        return False

    # Gradients are reusable (overwritten each backward pass)
    if purpose == "gradient":
        return True

    # Buffers without grad are reusable
    if purpose == "buffer" and not tensor.requires_grad:
        return True

    return False
