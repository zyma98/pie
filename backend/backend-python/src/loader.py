"""
Model loading for PIE backend.

This module handles the messy file I/O, ztensor vs safetensors,
TOML parsing, and model initialization.
"""

from __future__ import annotations

import tomllib
from contextlib import ExitStack
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from tqdm import tqdm
import ztensor
import safetensors

from .model import llama3, Config as ModelConfig

if TYPE_CHECKING:
    from .config import RuntimeConfig


class ModelLoader:
    """
    Handles model loading, weight I/O, and initialization.
    
    This separates the loading concerns from the runtime orchestration.
    """

    def __init__(self, config: "RuntimeConfig"):
        """
        Initialize the model loader.
        
        Args:
            config: Runtime configuration
        """
        self.config = config
        self.info: dict = {}

    def load(self) -> tuple[llama3.ForwardPass, list[torch.Tensor], llama3.Spec, dict]:
        """
        Load the model and return all components.
        
        Returns:
            Tuple of (forward_pass, kv_cache, model_spec, model_info)
        """
        # Load model info from TOML
        self.info = self._load_toml()
        arch_type = self.info["architecture"]["type"]

        # Normalize architecture fields
        normalized_arch = self._normalize_arch_fields(self.info["architecture"])

        # Load parameters and create forward pass
        match arch_type:
            case "llama3" | "l4ma":
                engine, kv_cache, model_spec = self._load_llama3(normalized_arch)
            case _:
                raise ValueError(f"Unsupported architecture type: {arch_type}")

        return engine, kv_cache, model_spec, self.info

    def _load_toml(self) -> dict:
        """
        Load model metadata from TOML file.
        
        Returns:
            Parsed TOML dictionary
        """
        # Try model subdirectory first
        model_info_path = (
            Path(self.config.cache_dir) / self.config.model / f"{self.config.model}.toml"
        )
        if not model_info_path.exists():
            # Fallback to cache_dir root
            model_info_path = Path(self.config.cache_dir) / f"{self.config.model}.toml"

        if not model_info_path.exists():
            raise ValueError(
                f'Metadata file for model "{self.config.model}" not found. '
                f"Expected: {self.config.cache_dir}/{self.config.model}/{self.config.model}.toml"
            )

        with open(model_info_path, "rb") as f:
            return tomllib.load(f)

    def _normalize_arch_fields(self, arch: dict) -> dict:
        """
        Normalize YAML/TOML field names to match Spec.from_dict expectations.
        
        Args:
            arch: Raw architecture dictionary from TOML
            
        Returns:
            Normalized architecture dictionary
        """
        normalized = dict(arch)

        # Map YAML field names -> expected names
        field_map = {
            "head_dim": "head_size",
            "num_heads": "num_query_heads",
            "num_heads_kv": "num_key_value_heads",
            "high_freq_factor": "high_frequency_factor",
            "low_freq_factor": "low_frequency_factor",
        }

        # Normalize top-level fields
        for old, new in field_map.items():
            if old in normalized and new not in normalized:
                normalized[new] = normalized.pop(old)

        # Normalize rope subfields
        if "rope" in normalized:
            rope = dict(normalized["rope"])
            for old, new in field_map.items():
                if old in rope and new not in rope:
                    rope[new] = rope.pop(old)
            # Add rope.factor default if missing
            if "factor" not in rope:
                rope["factor"] = 1.0
            normalized["rope"] = rope

        # Add missing fields with defaults
        if "rms_norm_eps" not in normalized:
            normalized["rms_norm_eps"] = 1e-5

        # Get vocab_size from tokenizer section if not in architecture
        if "vocab_size" not in normalized and "tokenizer" in self.info:
            tokenizer = self.info["tokenizer"]
            if "vocab_size" in tokenizer:
                normalized["vocab_size"] = tokenizer["vocab_size"]

        return normalized

    def _create_model_config(self, arch: dict) -> ModelConfig:
        """
        Create a model.Config object from RuntimeConfig and architecture.
        
        Args:
            arch: Normalized architecture dictionary
            
        Returns:
            Model configuration object
        """
        return ModelConfig(
            devices=self.config.device,
            rank=0,  # Single-rank for now
            activation_dtype=self.config.activation_dtype,
            weight_dtype=self.config.weight_dtype,
            kv_page_size=self.config.kv_page_size,
            max_dist_size=self.config.max_dist_size,
            max_num_embeds=self.config.max_num_embeds,
            max_batch_tokens=self.config.max_batch_tokens,
            max_num_adapters=self.config.max_num_adapters,
            max_adapter_rank=self.config.max_adapter_rank,
            max_num_kv_pages=self.config.max_num_kv_pages,
            mem_utilization=self.config.mem_utilization,
        )

    def _load_llama3(
        self, normalized_arch: dict
    ) -> tuple[llama3.ForwardPass, list[torch.Tensor], llama3.Spec]:
        """
        Load Llama3-style model.
        
        Args:
            normalized_arch: Normalized architecture dictionary
            
        Returns:
            Tuple of (forward_pass, kv_cache, model_spec)
        """
        # Determine path to model weight files
        model_dir = Path(self.config.cache_dir) / self.config.model
        if not model_dir.exists():
            # Fallback: weights might be in cache_dir directly
            model_dir = Path(self.config.cache_dir)

        # Create model spec and config
        model_spec = llama3.Spec.from_dict(normalized_arch)
        model_config = self._create_model_config(normalized_arch)

        # Load weights
        with ExitStack() as stack:
            readers: dict[str, object] = {}

            # Build tensor name -> reader mapping
            param_files = self.info.get("parameters", [])
            for param_file in tqdm(
                param_files, desc="Scanning tensor files", unit="files"
            ):
                param_path = model_dir / param_file

                if param_path.suffix == ".zt":
                    f = stack.enter_context(ztensor.Reader(str(param_path)))
                    names = f.get_tensor_names()
                elif param_path.suffix == ".safetensors":
                    f = stack.enter_context(
                        safetensors.safe_open(
                            str(param_path), framework="pt", device="cpu"
                        )
                    )
                    names = list(f.keys())
                else:
                    continue

                for n in names:
                    readers[n] = f

            def reader(
                name: str, *, expected_shape: tuple[int, ...] | None = None
            ) -> torch.Tensor:
                f = readers.get(name)
                if f is None:
                    raise KeyError(f"Tensor '{name}' not found")

                # ztensor vs safetensors
                t = (
                    f.read_tensor(name, to="torch")  # ztensor
                    if hasattr(f, "read_tensor")
                    else f.get_tensor(name)  # safetensors
                )

                if expected_shape is not None and tuple(t.shape) != tuple(
                    expected_shape
                ):
                    raise ValueError(
                        f"{name} has shape {tuple(t.shape)}, expected {tuple(expected_shape)}"
                    )
                return t

            # Load weights using schema
            weights = llama3.load_weights(
                model_spec,
                model_config,
                reader,
            )

            # Create forward pass with weights
            forward_pass = llama3.ForwardPass(
                model_spec,
                model_config,
                weights,
            )

            # Create KV cache
            kv_cache = llama3.create_kv_cache(model_spec, model_config)

        return forward_pass, kv_cache, model_spec
