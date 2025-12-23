"""
WeightSchema: A declarative API for model weight loading.

This module provides a "manifest" style API for defining how model weights
are loaded, fused, sharded, and transformed. The schema becomes a readable
specification of the physical-to-logical tensor mapping.

Example:
    schema = (
        Schema("llama3")
        .define("token_embeds",
            Source("model.embed_tokens.weight")
            .shard(Sharding.ROW))
        .define("layers.*.attn.qkv",
            Source.fuse([
                "model.layers.*.self_attn.q_proj.weight",
                "model.layers.*.self_attn.k_proj.weight",
                "model.layers.*.self_attn.v_proj.weight",
            ], dim=0)
            .shard(Sharding.COLUMN)
            .quantize())
    )

    weights = schema.load(reader, config, num_layers=32)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Any

import torch


# Sharding strategy strings: None, 'column', or 'row'
ShardingStrategy = str | None


@dataclass
class Source:
    """
    Defines a source tensor or fusion of multiple source tensors.
    
    Use method chaining to apply transforms:
        Source("weight.name").shard(Sharding.ROW).quantize()
    """
    
    _patterns: list[str]
    _fuse_dim: int | None = None
    _sharding: str | None = None  # None, 'column', or 'row'
    _should_quantize: bool = False
    _expected_shapes: list[tuple[int, ...] | None] | None = None
    
    def __init__(self, pattern: str):
        """Create a source from a single tensor pattern."""
        self._patterns = [pattern]
        self._fuse_dim = None
        self._sharding = None
        self._should_quantize = False
        self._expected_shapes = None
    
    @classmethod
    def fuse(cls, patterns: list[str], dim: int = 0) -> Source:
        """
        Create a fused source from multiple tensor patterns.
        
        Args:
            patterns: List of tensor name patterns to fuse
            dim: Dimension along which to concatenate (default: 0)
        
        Returns:
            A new Source configured for fusion
        """
        source = cls.__new__(cls)
        source._patterns = patterns
        source._fuse_dim = dim
        source._sharding = None
        source._should_quantize = False
        source._expected_shapes = None
        return source
    
    def shard(self, strategy: str) -> Source:
        """Set sharding strategy: 'column' or 'row'."""
        if strategy not in ('column', 'row'):
            raise ValueError(f"Invalid sharding strategy: {strategy}. Use 'column' or 'row'.")
        self._sharding = strategy
        return self
    
    def quantize(self) -> Source:
        """Enable quantization for this weight."""
        self._should_quantize = True
        return self
    
    def expect_shapes(self, shapes: list[tuple[int, ...] | None]) -> Source:
        """Set expected shapes for validation (optional)."""
        self._expected_shapes = shapes
        return self
    
    @property
    def is_fused(self) -> bool:
        return self._fuse_dim is not None
    
    @property
    def patterns(self) -> list[str]:
        return self._patterns
    
    @property
    def fuse_dim(self) -> int | None:
        return self._fuse_dim
    
    @property
    def sharding(self) -> str | None:
        return self._sharding
    
    @property
    def should_quantize(self) -> bool:
        return self._should_quantize


@dataclass
class Definition:
    """A named weight definition: logical name -> Source with transforms."""
    
    name: str  # Logical name, may contain '*' for layer patterns
    source: Source
    
    def has_layer_pattern(self) -> bool:
        """Check if this definition uses layer patterns (*)."""
        return "*" in self.name
    
    def expand_for_layer(self, layer_idx: int) -> str:
        """Expand the logical name for a specific layer."""
        return self.name.replace("*", str(layer_idx))
    
    def expand_source_for_layer(self, layer_idx: int) -> list[str]:
        """Expand source patterns for a specific layer."""
        return [p.replace("*", str(layer_idx)) for p in self.source.patterns]


class WeightStore:
    """
    Container for loaded weights, accessible by logical name.
    """
    
    def __init__(self):
        self._weights: dict[str, torch.Tensor] = {}
    
    def put(self, name: str, tensor: torch.Tensor) -> None:
        """Store a tensor by logical name."""
        self._weights[name] = tensor
    
    def get(self, name: str) -> torch.Tensor:
        """Retrieve a tensor by logical name."""
        if name not in self._weights:
            raise KeyError(f"Weight '{name}' not found in store")
        return self._weights[name]
    
    def get_list(self, pattern: str, count: int) -> list[torch.Tensor]:
        """
        Retrieve a list of tensors matching a pattern.
        
        Args:
            pattern: Pattern with '*' placeholder (e.g., "layers.*.proj_qkv")
            count: Number of items to retrieve
        
        Returns:
            List of tensors for indices 0..count-1
        """
        return [self.get(pattern.replace("*", str(i))) for i in range(count)]
    
    def keys(self) -> list[str]:
        """List all stored weight names."""
        return list(self._weights.keys())
    
    def __contains__(self, name: str) -> bool:
        return name in self._weights
    
    def __len__(self) -> int:
        return len(self._weights)


ReaderFn = Callable[[str], torch.Tensor]


@dataclass
class LoadConfig:
    """Configuration for weight loading."""
    
    device: torch.device
    world_size: int = 1
    rank: int = 0
    quantization: Any = None  # torchao quantization config
    
    @property
    def needs_sharding(self) -> bool:
        return self.world_size > 1


class Schema:
    """
    Schema definition for weight loading.
    
    Use method chaining to define weights:
        schema = Schema("model_name").define("name", Source(...))
    """
    
    def __init__(self, name: str):
        self.name = name
        self._definitions: list[Definition] = []
    
    def define(self, name: str, source: Source) -> Schema:
        """
        Define a logical weight with its source.
        
        Args:
            name: Logical name (may contain '*' for per-layer weights)
            source: Source configuration
        
        Returns:
            self for method chaining
        """
        self._definitions.append(Definition(name=name, source=source))
        return self
    
    def load(
        self,
        reader: ReaderFn,
        config: LoadConfig,
        num_layers: int = 0,
    ) -> WeightStore:
        """
        Load all weights according to the schema.
        
        Args:
            reader: Function to read tensors by name
            config: Loading configuration (device, sharding, quantization)
            num_layers: Number of layers (for expanding '*' patterns)
        
        Returns:
            WeightStore with all loaded weights
        """
        store = WeightStore()
        
        for defn in self._definitions:
            if defn.has_layer_pattern():
                # Expand for each layer
                for layer_idx in range(num_layers):
                    logical_name = defn.expand_for_layer(layer_idx)
                    physical_names = defn.expand_source_for_layer(layer_idx)
                    tensor = self._load_single(
                        reader, config, defn.source, physical_names
                    )
                    store.put(logical_name, tensor)
            else:
                # Single tensor (not per-layer)
                tensor = self._load_single(
                    reader, config, defn.source, defn.source.patterns
                )
                store.put(defn.name, tensor)
        
        return store
    
    def _load_single(
        self,
        reader: ReaderFn,
        config: LoadConfig,
        source: Source,
        physical_names: list[str],
    ) -> torch.Tensor:
        """Load a single (possibly fused) tensor with transforms applied."""
        # Read tensors
        tensors = [reader(name) for name in physical_names]
        
        # Fuse if needed
        if source.is_fused:
            tensor = torch.cat(tensors, dim=source.fuse_dim)
        else:
            tensor = tensors[0]
        
        # Apply sharding
        if config.needs_sharding and source.sharding is not None:
            dim = 1 if source.sharding == 'row' else 0
            
            # Validate dimension is divisible by world_size
            if tensor.shape[dim] % config.world_size != 0:
                raise ValueError(
                    f"Cannot shard tensor of shape {tuple(tensor.shape)} along dim={dim}: "
                    f"dimension size {tensor.shape[dim]} is not divisible by world_size={config.world_size}"
                )
            
            tensor = torch.chunk(tensor.contiguous(), config.world_size, dim=dim)[config.rank]
        
        # Apply quantization (lazy import to avoid dependency issues)
        if source.should_quantize and config.quantization is not None:
            from .quantization import quantize
            tensor = quantize(tensor, config.quantization)
        
        # Move to device
        tensor = tensor.to(config.device)
        
        return tensor
