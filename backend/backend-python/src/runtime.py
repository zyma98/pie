from __future__ import annotations
from dataclasses import dataclass, asdict, field

import base64
import tomllib
import yaml
from contextlib import ExitStack, closing
from pathlib import Path
from typing import Callable

import numpy as np
from tqdm import tqdm
import torch
import ztensor
import safetensors

from .model import llama3
from .adapter import AdapterSubpass
from .utils import resolve_cache_dir
from . import message


@dataclass
class RuntimeConfig:
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
    dtype: torch.dtype

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
        dtype: str = "bfloat16",
        enable_profiling: bool = False,
    ) -> "RuntimeConfig":
        """
        Factory method to build a validated and resolved RuntimeConfig.
        This replaces the original `build_config` logic.
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

        # Resolve dtype
        resolved_dtype = getattr(torch, dtype, torch.bfloat16)

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
            dtype=resolved_dtype,
        )

    def print(self) -> None:
        """
        Utility to print configuration in a consistent format.
        This replaces the original `print_config` function.
        """
        print("--- Configuration ---")
        # asdict() conveniently converts the dataclass to a dict
        for key, value in asdict(self).items():
            print(f"{key}: {value}")
        print("----------------------")


class LoadedModel:

    config: ModelConfig
    param: object
    buffer: object
    forward_pass: object
    batch: SyncBatch


class Runtime:

    config: ModelConfig
    rank: int

    model_config: llama3.Config 
    model_param: llama3.Param
    model_pass: llama3.ForwardPass

    adapter_at_layer: list[tuple[torch.Tensor, torch.Tensor]]
    kv_cache_at_layer: list[tuple[torch.Tensor, torch.Tensor]]

    batch: SyncBatch
    adapters: dict

    def __init__(self, config: RuntimeConfig):

        self.config = config
        self.adapters = {}

        # Try both path formats: {cache_dir}/{model}/{model}.yaml and {cache_dir}/{model}.toml
        model_info_path_yaml = Path(config.cache_dir) / config.model / f"{config.model}.yaml"
        model_info_path_toml = Path(config.cache_dir) / f"{config.model}.toml"

        if model_info_path_yaml.exists():
            with open(model_info_path_yaml, "r") as f:
                self.info = yaml.safe_load(f)
        elif model_info_path_toml.exists():
            with open(model_info_path_toml, "rb") as f:
                self.info = tomllib.load(f)
        else:
            raise ValueError(
                f'Metadata file for model "{config.model}" not found. '
                f'Tried: {model_info_path_yaml}, {model_info_path_toml}'
            )

        self.type = self.info["architecture"]["type"]

        # Load model parameters
        self._load_params()

        # Initialize KV cache and adapter states
        self._init_states()

    def _load_params(self) -> None:
        """Load model parameters from weight files and create ForwardPass."""
        # Determine path to model weight files
        model_dir = Path(self.config.cache_dir) / self.config.model
        if not model_dir.exists():
            # Fallback: weights might be in cache_dir directly
            model_dir = Path(self.config.cache_dir)

        # Normalize architecture field names for spec creation
        arch = self.info["architecture"]
        normalized_arch = self._normalize_arch_fields(arch)

        with ExitStack() as stack:
            readers: dict[str, object] = {}

            # Build tensor name -> reader mapping
            param_files = self.info.get("parameters", [])
            for param_file in tqdm(
                param_files, desc="Scanning tensor files", unit="files"
            ):
                param_path = model_dir / param_file

                if param_path.suffix == ".zt":
                    f = stack.enter_context(
                        ztensor.Reader(str(param_path))
                    )
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

            # Create model-specific spec, param, and forward pass
            match self.type:
                case "llama3" | "l4ma":  # Support both naming conventions
                    # Create model config for llama3.Buffer
                    model_config = self._create_model_config(normalized_arch)

                    self.model_spec = llama3.Spec.from_dict(normalized_arch)
                    self.model_param = llama3.Param.from_reader(
                        self.model_spec,
                        model_config,
                        reader,
                    )
                    self.forward_pass = llama3.ForwardPass(self.config.device[0])
                    self.kv_cache_at_layer = llama3.Buffer.from_config(
                        self.model_spec, model_config
                    ).kv_cache
                case _:
                    raise ValueError(f"Unsupported architecture type: {self.type}")

    def _normalize_arch_fields(self, arch: dict) -> dict:
        """Normalize YAML field names to match Spec.from_dict expectations."""
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

    def _create_model_config(self, arch: dict):
        """Create a model.Config object from RuntimeConfig and architecture."""
        from .model import Config

        return Config(
            devices=self.config.device,
            rank=0,  # Single-rank for now
            dtype=self.config.dtype,
            quantization=None,  # TODO: support quantization
            kv_page_size=self.config.kv_page_size,
            max_dist_size=self.config.max_dist_size,
            max_num_embeds=self.config.max_num_embeds,
            max_batch_tokens=self.config.max_batch_tokens,
            max_num_adapters=self.config.max_num_adapters,
            max_adapter_rank=self.config.max_adapter_rank,
            max_num_kv_pages=self.config.max_num_kv_pages,
            mem_utilization=self.config.mem_utilization,
        )


    def _init_states(self):
        """Initialize adapter states. KV cache is created in _load_params."""
        device = self.config.device[0]

        # Only create adapter states if model_spec exists (called after _load_params)
        if not hasattr(self, 'model_spec'):
            return

        self.adapter_at_layer = [
            (
                torch.zeros(
                    (
                        self.config.max_num_adapters,
                        self.config.max_adapter_rank * 3,
                        self.model_spec.dim_hidden,
                    ),
                    dtype=self.config.dtype,
                    device=device,
                ),
                torch.zeros(
                    (
                        self.config.max_num_adapters,
                        self.model_spec.dim_head
                        * (
                            self.model_spec.num_q_heads
                            + self.model_spec.num_kv_heads * 2
                        ),
                        self.config.max_adapter_rank,
                    ),
                    dtype=self.config.dtype,
                    device=device,
                ),
            )
            for _ in range(self.model_spec.num_layers)
        ]

    def get_metadata(self) -> dict:
        return {
            "name": self.info.get("name", self.config.model),
            "description": self.info.get("description", ""),
            "version": self.info.get("version", "1.0.0"),
        }

    def get_chat_template(self) -> dict:
        template = self.info.get("template", {})
        return {
            "template_type": template.get("type", "none"),
            "template_content": template.get("content", ""),
            "stop_tokens": template.get("stop_tokens", []),
        }

    def get_tokenizer(self) -> dict:
        tokenizer_info = self.info.get("tokenizer", {})
        model_dir = Path(self.config.cache_dir) / self.config.model

        # Get vocab file path
        vocab_filename = tokenizer_info.get("vocab") or tokenizer_info.get("vocabulary_file")
        if not vocab_filename:
            # Return minimal tokenizer info if no vocab file
            return {
                "type": tokenizer_info.get("type", "bpe"),
                "num_vocab": tokenizer_info.get("vocab_size", self.model_param.embed_token.shape[0]),
                "merge_table": {},
                "split_regex": tokenizer_info.get("split_regex", ""),
                "special_tokens": tokenizer_info.get("special_tokens", {}),
                "escape_non_printable": tokenizer_info.get("escape_non_printable", False),
            }

        vocab_file_path = model_dir / vocab_filename
        merge_rules: dict[int, bytes] = {}

        if vocab_file_path.exists():
            with open(vocab_file_path, "r", encoding="utf-8") as f:
                for line_number, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) != 2:
                        continue
                    b64_token, rank_str = parts
                    try:
                        decoded_token = base64.b64decode(b64_token)
                        rank = int(rank_str)
                        merge_rules[rank] = decoded_token
                    except (ValueError, TypeError):
                        continue

        return {
            "type": tokenizer_info.get("type", "bpe"),
            "num_vocab": tokenizer_info.get("vocab_size", self.model_param.embed_token.shape[0]),
            "merge_table": merge_rules,
            "split_regex": tokenizer_info.get("split_regex", ""),
            "special_tokens": tokenizer_info.get("special_tokens", {}),
            "escape_non_printable": tokenizer_info.get("escape_non_printable", False),
        }

    # ========================================================================
    # Service Protocol Implementation
    # ========================================================================

    def handshake(
        self, reqs: list[message.HandshakeRequest]
    ) -> list[message.HandshakeResponse]:
        """Handle handshake requests returning model and tokenizer info."""
        metadata = self.get_metadata()
        template = self.get_chat_template()
        tokenizer = self.get_tokenizer()

        responses = []
        for _ in reqs:
            resp = message.HandshakeResponse(
                version=metadata.get("version", "1.0.0"),
                model_name=metadata["name"],
                model_traits=[],  # TODO: populate traits
                model_description=metadata.get("description", ""),
                prompt_template=template["template_content"],
                prompt_template_type=template["template_type"],
                prompt_stop_tokens=template["stop_tokens"],
                kv_page_size=self.config.kv_page_size,
                max_batch_tokens=self.config.max_batch_tokens or 10240,
                resources={
                    0: self.config.max_num_kv_pages or 0,
                    1: self.config.max_num_embeds,
                    2: self.config.max_num_adapters,
                },
                tokenizer_num_vocab=tokenizer["num_vocab"],
                tokenizer_merge_table=tokenizer["merge_table"],
                tokenizer_special_tokens=tokenizer["special_tokens"],
                tokenizer_split_regex=tokenizer["split_regex"],
                tokenizer_escape_non_printable=tokenizer["escape_non_printable"],
            )
            responses.append(resp)
        return responses

    def heartbeat(
        self, reqs: list[message.HeartbeatRequest]
    ) -> list[message.HeartbeatResponse]:
        """Handle heartbeat keepalive requests."""
        return [message.HeartbeatResponse() for _ in reqs]

    def query(
        self, reqs: list[message.QueryRequest]
    ) -> list[message.QueryResponse]:
        """Handle query requests."""
        responses = []
        for req in reqs:
            value = "unknown query"
            match req.query:
                case "ping":
                    value = "pong"
            responses.append(message.QueryResponse(value=value))
        return responses

    def forward_pass(
        self, reqs: list[message.ForwardPassRequest]
    ) -> list[message.ForwardPassResponse]:
        """Handle batched forward pass inference requests."""
        # Accumulate requests into the batch
        for req in reqs:
            self._add_forward_pass_request(req)
        # Execute the batch and return responses
        return self.block_on_forward_pass()

    def embed_image(self, reqs: list[message.EmbedImageRequest]) -> None:
        """Handle image embedding requests."""
        # TODO: implement image embedding
        pass

    def initialize_adapter(
        self, reqs: list[message.InitializeAdapterRequest]
    ) -> None:
        """Initialize adapter functionality."""
        for req in reqs:
            self._initialize_adapter(
                adapter_ptr=req.adapter_ptr,
                rank=req.rank,
                alpha=req.alpha,
                population_size=req.population_size,
                mu_fraction=req.mu_fraction,
                initial_sigma=req.initial_sigma,
            )

    def update_adapter(self, reqs: list[message.UpdateAdapterRequest]) -> None:
        """Update adapter parameters."""
        for req in reqs:
            self._update_adapter(
                adapter_ptr=req.adapter_ptr,
                scores=req.scores,
                seeds=req.seeds,
                max_sigma=req.max_sigma,
            )

    def upload_adapter(self, reqs: list[message.UploadAdapterRequest]) -> None:
        """Upload adapter weights."""
        # TODO: implement adapter upload
        pass

    def download_adapter(
        self, reqs: list[message.DownloadAdapterRequest]
    ) -> list[message.DownloadAdapterResponse]:
        """Download adapter weights."""
        # TODO: implement adapter download
        return [message.DownloadAdapterResponse(adapter_data=b"") for _ in reqs]

    # ========================================================================
    # Internal Adapter Methods (renamed from public methods)
    # ========================================================================

    @torch.inference_mode()
    def _initialize_adapter(
        self,
        adapter_ptr: int,
        rank: int,
        alpha: float,
        population_size: int,
        mu_fraction: float,
        initial_sigma: float,
    ):
        raise NotImplementedError

    @torch.inference_mode()
    def _update_adapter(
        self,
        adapter_ptr: int,
        scores: list[float],
        seeds: list[int],
        max_sigma: float,
    ):
        raise NotImplementedError

    @torch.inference_mode()
    def _add_forward_pass_request(
        self,
        input_tokens: list[int],
        input_token_positions: list[int],
        adapter: int | None,
        adapter_seed: int | None,
        mask: list[list[int]],
        kv_page_ptrs: list[int],
        kv_page_last_len: int,
        output_token_indices: list[int],
        output_token_samplers: list[dict],
        output_embed_ptrs: list[int],
        output_embed_indices: list[int],
    ):

        # init batch if not present
        if self.batch is None:
            self.batch = SyncBatch()
        batch = self.batch

        # Handle adapter information
        if adapter is not None and adapter in self.adapters:
            seed = adapter_seed if adapter_seed is not None else 0
            batch.seeds.extend([seed] * len(input_tokens))
            batch.adapter_indices.append(adapter)
            batch.adapter_subpass_needed = True

        # Handle KV cache pages
        batch.kv_page_indices.extend(kv_page_ptrs)
        batch.kv_page_indptr.append(len(batch.kv_page_indices))
        batch.kv_last_page_lengths.append(kv_page_last_len or 0)

        # Handle output mappings for embeddings that need to be stored
        if len(output_embed_indices) != len(output_embed_ptrs):
            raise ValueError(
                f"Mismatch between output_embed_indices length ({len(output_embed_indices)}) "
                f"and output_embed_ptrs length ({len(output_embed_ptrs)})"
            )
        for token_idx, storage_ptr in zip(output_embed_indices, output_embed_ptrs):
            batch.indices_for_embed_storage.append(
                token_idx + batch.total_tokens_in_batch
            )
            batch.embed_storage_pointers.append(storage_ptr)

        # Handle output mappings for tokens requiring logits.
        for token_idx in output_token_indices:
            batch.indices_for_logits.append(token_idx + batch.total_tokens_in_batch)

        # Extract sampler configurations.
        # sampler_idx=0 is for distributions, existing samplers are shifted by +1.
        for sampler_config in output_token_samplers:
            params = {}
            sampler_idx = sampler_config["sampler"]
            batch.sampler_type.append(sampler_idx)

            if sampler_idx == 0:
                params["top_k"] = min(
                    sampler_config.get("top_k", self.config.max_dist_size),
                    self.config.max_dist_size,
                )
            else:
                params["top_k"] = sampler_config.get("top_k", 0)
                params["top_p"] = sampler_config.get("top_p", 1.0)
                params["min_p"] = sampler_config.get("min_p", 0.0)

            params["temperature"] = sampler_config.get("temperature", 1.0)
            batch.sampler_params.append(params)

        # Handle input tokens and positions
        batch.batch_token_ids.extend(input_tokens)
        batch.batch_position_ids.extend(input_token_positions)
        batch.total_tokens_in_batch += len(input_tokens)
        batch.qo_indptr.append(batch.total_tokens_in_batch)

        if len(input_tokens) > 1:
            batch.single_token_inference_mode = False

        attention_mask = _generate_mask_for_request(
            input_tokens,
            mask,
            kv_page_ptrs,
            kv_page_last_len,
            self.config.kv_page_size,
        )
        batch.attention_masks.append(attention_mask)

    @torch.inference_mode()
    def block_on_forward_pass(self):
        """Finalizes batch preparation, creating tensors and the adapter subpass."""
        batch = self.batch
        device = self.config.device[self.rank]

        adapter_subpass = None
        if batch.adapter_subpass_needed:
            seeds_tensor = torch.as_tensor(batch.seeds, device=device, dtype=torch.long)
            adapter_subpass = AdapterSubpass(
                adapter_at_layer=self.adapter_at_layer,
                adapter_indices=batch.adapter_indices,
                adapter_extras=self.adapters,
                rand_seeds=seeds_tensor,
                qo_indptr=batch.qo_indptr,
            )

        batched_attention_mask = (
            np.concatenate(batch.attention_masks)
            if batch.attention_masks
            else np.array([], dtype=np.bool_)
        )
        token_ids_tensor = torch.as_tensor(
            batch.batch_token_ids, device=device, dtype=torch.int32
        )

        input_embeds = self.model_pass.embed_tokens(
            param=self.model_param,
            token_ids=token_ids_tensor,
        )

        hidden_states = self.model_pass.transform(
            param=self.model_param,
            input_embeds=input_embeds,
            position_ids=torch.as_tensor(
                batch.batch_position_ids, device=device, dtype=torch.int32
            ),
            qo_indptr=torch.as_tensor(
                batch.qo_indptr, device=device, dtype=torch.int32
            ),
            kv_cache_at_layer=self.kv_cache_at_layer,
            kv_page_indices=torch.as_tensor(
                batch.kv_page_indptr, device=device, dtype=torch.int32
            ),
            kv_page_indptr=torch.as_tensor(
                batch.kv_page_indptr, device=device, dtype=torch.int32
            ),
            kv_last_page_lens=torch.as_tensor(
                batch.kv_last_page_lengths, device=device, dtype=torch.int32
            ),
            custom_mask=torch.as_tensor(
                batched_attention_mask, device=device, dtype=torch.bool
            ),
            single_token_inference_mode=batch.single_token_inference_mode,
            adapter_subpass=adapter_subpass,
        )

        # --------------
        # Apply temperature scaling to all logits
        temperatures = torch.tensor(
            [p["temperature"] for p in batch.sampler_params],
            device=device,
            dtype=logits.dtype,
        ).unsqueeze(1)

        # Group requests by sampler type for efficient batch processing
        sampler_groups = {}
        for i, sampler_idx in enumerate(batch.sampler_type):
            if sampler_idx not in sampler_groups:
                sampler_groups[sampler_idx] = []
            sampler_groups[sampler_idx].append(i)

        sample_outputs, final_dists = self.model_pass.sample(
            param=self.model_param,
            hidden_states=hidden_states,
            indices_for_logits=batch.indices_for_logits,
            temperatures=temperatures,
            sampler_groups=sampler_groups,
            sampler_params=batch.sampler_params,
        )

        # --samling

        responses = []
        cursor = 0
        for req in self.requests:
            output_token_indices = req.output_token_indices or []
            num_outputs = len(output_token_indices)
            request_dists = []
            request_tokens = []

            # Iterate through the slice of results belonging to this request
            for i in range(cursor, cursor + num_outputs):
                if self.sampler_type[i] == 0:  # This was a distribution request
                    if final_dists[i] is not None:
                        request_dists.append(final_dists[i])
                else:  # This was a sampling request
                    request_tokens.append(final_tokens_tensor[i].item())

            responses.append(
                message.ForwardPassResponse(dists=request_dists, tokens=request_tokens)
            )
            cursor += num_outputs

        return responses

    @torch.inference_mode()
    def upload_adapter(
        self,
        adapter_ptr: int,
        scores: list[float],
        seeds: list[int],
        max_sigma: float,
    ):
        raise NotImplementedError

    @torch.inference_mode()
    def download_adapter(
        self, adapter_data: bytes
    ) -> list[message.DownloadAdapterResponse]:
        raise NotImplementedError


@dataclass
class SyncBatch:
    adapter_indices: list[int] = field(default_factory=list)
    seeds: list[int] = field(default_factory=list)
    kv_page_indices: list[int] = field(default_factory=list)
    kv_page_indptr: list[int] = field(default_factory=list)
    kv_last_page_lengths: list[int] = field(default_factory=list)
    qo_indptr: list[int] = field(default_factory=list)
    attention_masks: list[np.ndarray] = field(default_factory=list)
    batch_token_ids: list[int] = field(default_factory=list)
    batch_position_ids: list[int] = field(default_factory=list)

    # tracking states
    total_tokens_in_batch: int = 0
    single_token_inference_mode: bool = True
    adapter_subpass_needed: bool = False

    # Output mapping for all logit-based operations (dists and sampling)
    indices_for_logits: list[int] = field(default_factory=list)
    indices_for_embed_storage: list[int] = field(default_factory=list)
    embed_storage_pointers: list[int] = field(default_factory=list)

    # sampler type and consolidated parameters
    sampler_type: list[int] = field(default_factory=list)
    sampler_params: list[dict] = field(default_factory=list)


def _generate_mask_for_request(
    input_tokens: list[int],
    mask: list[list[int]],
    kv_page_ptrs: list[int],
    kv_page_last_len: int,
    kv_page_size: int,
) -> np.ndarray:
    """Generates the custom attention mask for a single request."""
    if len(mask) != len(input_tokens):
        raise ValueError(
            f"Mismatch between number of masks ({len(mask)}) and "
            f"input tokens ({len(input_tokens)})."
        )

    # Ensure we have at least one page for proper computation
    if len(kv_page_ptrs) >= 1:
        sequence_length = kv_page_size * (len(kv_page_ptrs) - 1) + kv_page_last_len
    else:
        sequence_length = kv_page_last_len

    # Validate sequence_length is sufficient for input tokens
    input_token_count = len(input_tokens)
    if sequence_length < input_token_count:
        raise ValueError(
            f"Insufficient sequence length ({sequence_length}) for input tokens "
            f"({input_token_count}). Sequence length must be at least equal to "
            f"the number of input tokens."
        )

    context_length = sequence_length - input_token_count

    request_attention_mask = np.zeros(
        (len(input_tokens), sequence_length), dtype=np.bool_
    )
    for i, brle_buffer in enumerate(mask):
        decoded_mask = _decode_brle(brle_buffer)
        expected_len = context_length + i + 1
        if len(decoded_mask) != expected_len:
            raise ValueError(
                f"Decoded mask for token {i} has length {len(decoded_mask)}, "
                f"but expected {expected_len}"
            )
        request_attention_mask[i, :expected_len] = decoded_mask

    return request_attention_mask.flatten()


def _decode_brle(brle_buffer: list[int]) -> np.ndarray:
    """
    Decodes a Binary Run-Length Encoded buffer into a boolean numpy array.
    The format assumes alternating runs of False and True, starting with False.
    """
    if not brle_buffer:
        return np.array([], dtype=bool)

    total_size = sum(brle_buffer)
    if total_size == 0:
        return np.array([], dtype=bool)

    decoded_array = np.empty(total_size, dtype=bool)
    current_pos = 0
    value = True  # In attention masking, True means attend.
    for run_len in brle_buffer:
        if run_len > 0:
            decoded_array[current_pos : current_pos + run_len] = value
        current_pos += run_len
        value = not value  # Flip value for the next run
    return decoded_array
