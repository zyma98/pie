"""
Runtime configuration for PIE backend.

This module contains the RuntimeConfig dataclass that encapsulates all
configuration options for the inference runtime.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import torch

from .utils import resolve_cache_dir


@dataclass
class RuntimeConfig:
    """Configuration for the PIE inference runtime."""

    model: str
    cache_dir: str
    kv_page_size: int
    max_dist_size: int
    max_num_embeds: int
    max_batch_tokens: int | None
    max_num_adapters: int
    max_adapter_rank: int
    max_num_kv_pages: int | None
    mem_utilization: float

    device: list[torch.device]
    rank: int
    activation_dtype: torch.dtype
    weight_dtype: str | None  # None means same as activation_dtype (no quantization)

    @classmethod
    def from_args(
        cls,
        model: str,
        cache_dir: str | None = None,
        kv_page_size: int = 16,
        max_dist_size: int = 64,
        max_num_embeds: int = 128,
        max_batch_tokens: int = 10240,
        max_num_adapters: int = 48,
        max_adapter_rank: int = 8,
        max_num_kv_pages: int | None = None,
        gpu_mem_headroom: float | None = None,
        device: str | None = None,
        activation_dtype: str = "bfloat16",
        weight_dtype: str | None = None,
        enable_profiling: bool = False,
    ) -> "RuntimeConfig":
        """
        Factory method to build a validated and resolved RuntimeConfig.

        Args:
            model: Model name or path
            cache_dir: Directory for model cache (resolved automatically if None)
            kv_page_size: Size of KV cache pages
            max_dist_size: Maximum distribution size for sampling
            max_num_embeds: Maximum number of embeddings
            max_batch_tokens: Maximum tokens per batch
            max_num_adapters: Maximum number of adapters
            max_adapter_rank: Maximum adapter rank
            max_num_kv_pages: Maximum KV pages (computed from memory if None)
            gpu_mem_headroom: GPU memory headroom fraction
            device: Device string (auto-detected if None)
            activation_dtype: Activation tensor dtype
            weight_dtype: Weight quantization type ("int4", "int8", "float8", or None)
            enable_profiling: Enable profiling (currently unused)

        Returns:
            Configured RuntimeConfig instance
        """
        # Resolution
        resolved_cache_dir = resolve_cache_dir(cache_dir)

        # Resolve device
        if device is None:
            if torch.cuda.is_available():
                resolved_device = torch.device("cuda:0")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                resolved_device = torch.device("mps")
            else:
                resolved_device = torch.device("cpu")
        else:
            resolved_device = torch.device(device)

        # Resolve activation dtype
        resolved_activation_dtype = getattr(torch, activation_dtype, torch.bfloat16)

        # Validate weight_dtype if specified
        valid_weight_dtypes = {"int4", "int8", "float8", None}
        if weight_dtype is not None and weight_dtype not in valid_weight_dtypes:
            raise ValueError(
                f"Invalid weight_dtype: '{weight_dtype}'. "
                f"Expected one of: 'int4', 'int8', 'float8', or None."
            )

        # Calculate mem_utilization from gpu_mem_headroom if provided
        mem_utilization = 1.0 - gpu_mem_headroom if gpu_mem_headroom else 0.9

        # Create the config instance
        return cls(
            model=model,
            cache_dir=resolved_cache_dir,
            kv_page_size=kv_page_size,
            max_dist_size=max_dist_size,
            max_num_embeds=max_num_embeds,
            max_batch_tokens=max_batch_tokens,
            max_num_adapters=max_num_adapters,
            max_adapter_rank=max_adapter_rank,
            max_num_kv_pages=max_num_kv_pages,
            mem_utilization=mem_utilization,
            device=[resolved_device],
            rank=0,
            activation_dtype=resolved_activation_dtype,
            weight_dtype=weight_dtype,
        )

    def print(self) -> None:
        """
        Utility to print configuration in a consistent format.
        """
        print("--- Configuration ---")
        for key, value in asdict(self).items():
            print(f"{key}: {value}")
        print("----------------------")
