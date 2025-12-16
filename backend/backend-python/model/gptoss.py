"""GPT OSS Large Language Model Architecture"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
import flashinfer as ops
from flashinfer.gemm import group_gemm_mxfp4_nt_groupwise
from flashinfer.fp4_quantization import _pad_scale_factors, get_fp4_quantization_module
from flashinfer.fp8_quantization import mxfp8_quantize
from flashinfer.utils import get_compute_capability
from adapter_utils import AdapterSubpass
from model.config import CommonArch, ModelConfig
from model.gptoss_utils import (
    FP4_VALUES,
    chunked_enumerate,
    GptOssRMSNorm,
    GptOssRotaryEmbedding,
)
from einops import einsum, rearrange


VERSION = "0.1.0"


@dataclass
class GptOssArch(CommonArch):
    """GPT OSS specific architecture configuration."""

    # MoE configuration
    num_experts: int
    experts_per_token: int

    # RoPE configuration
    rope_theta: float
    rope_scaling_factor: float
    rope_ntk_alpha: float
    rope_ntk_beta: float

    # Model specific parameters
    initial_context_length: int
    sliding_window: int
    swiglu_limit: float

    @staticmethod
    def from_config(cfg: ModelConfig) -> "GptOssArch":
        """Parse GPT OSS-specific architecture configuration."""
        # Get common architecture fields
        common_arch_dict = cfg.get_common_arch_dict()

        # Get all the fields for the architecture section to grab other
        # architecture-specific fields
        arch_dict = cfg.get_required_key(cfg.root, "architecture")

        # Get MoE configuration
        moe_dict = cfg.get_required_key(arch_dict, "moe")
        num_experts = cfg.get_required_key(moe_dict, "num_experts")
        experts_per_token = cfg.get_required_key(moe_dict, "experts_per_token")

        # Get RoPE configuration (GPT OSS uses YaRN-style RoPE)
        rope_dict = cfg.get_required_key(arch_dict, "rope")
        rope_theta = cfg.get_required_key(rope_dict, "theta")
        rope_scaling_factor = cfg.get_required_key(rope_dict, "scaling_factor")
        rope_ntk_alpha = cfg.get_required_key(rope_dict, "ntk_alpha")
        rope_ntk_beta = cfg.get_required_key(rope_dict, "ntk_beta")

        # Get model specific parameters
        initial_context_length = cfg.get_required_key(
            arch_dict, "initial_context_length"
        )
        sliding_window = cfg.get_required_key(arch_dict, "sliding_window")
        swiglu_limit = cfg.get_required_key(arch_dict, "swiglu_limit")

        return GptOssArch(
            # Common fields
            **common_arch_dict,
            # GPT OSS-specific fields
            num_experts=num_experts,
            experts_per_token=experts_per_token,
            rope_theta=rope_theta,
            rope_scaling_factor=rope_scaling_factor,
            rope_ntk_alpha=rope_ntk_alpha,
            rope_ntk_beta=rope_ntk_beta,
            initial_context_length=initial_context_length,
            sliding_window=sliding_window,
            swiglu_limit=swiglu_limit,
        )


def create_fusion_map(model: nn.Module):
    """
    Analyzes the model and creates a map for fusing weights and handling MXFP4 tensors.

    Returns:
        A dictionary mapping {
            fused_tensor_name: {"sources": [source_names], "dim": cat_dim, "type": type}
        }.
        For MXFP4 tensors, type is "mxfp4" and sources contains [blocks_name, scales_name].
        For fusion tensors, type is "fusion" and sources contains the tensors to concatenate.
        For regular tensors, type is "regular" and sources contains the single tensor name.
    """
    fusion_map = {}
    for name, module in model.named_modules():
        # --- Rule for GptOssAttention QKV Fusion ---
        if isinstance(module, GptOssAttention):
            # Handle weights
            target_w = f"{name}.qkv_proj.weight"
            sources_w = [
                f"{name}.q_proj.weight",
                f"{name}.k_proj.weight",
                f"{name}.v_proj.weight",
            ]
            fusion_map[target_w] = {"sources": sources_w, "dim": 0, "op": "fusion"}

            # Handle biases if they exist
            if module.qkv_proj.bias is not None:
                target_b = f"{name}.qkv_proj.bias"
                sources_b = [
                    f"{name}.q_proj.bias",
                    f"{name}.k_proj.bias",
                    f"{name}.v_proj.bias",
                ]
                fusion_map[target_b] = {
                    "sources": sources_b,
                    "dim": 0,
                    "op": "fusion",
                }

        # --- Rule for GptOssExperts MXFP4 Weights ---
        elif isinstance(module, GptOssExperts):
            # Handle gate_up_proj weights (MXFP4 format)
            # Weights are PRE-PADDED and scales are PRE-SWIZZLED at load time for efficiency
            target_blocks_gate_up = f"{name}.gate_up_proj_blocks"
            source_blocks_gate_up = f"{name}.gate_up_proj_blocks"
            fusion_map[target_blocks_gate_up] = {
                "sources": [source_blocks_gate_up],
                "op": "load_mxfp4_blocks_padded",
                "k_original": module.gate_up_k,
                "k_gemm": module.gate_up_k_gemm,
            }

            target_scales_gate_up = f"{name}.gate_up_proj_scales"
            source_scales_gate_up = f"{name}.gate_up_proj_scales"
            fusion_map[target_scales_gate_up] = {
                "sources": [source_scales_gate_up],
                "op": "load_mxfp4_scales_swizzled",
                "n": module.intermediate_size * 2,
                "n_padded": module.gate_up_n_padded,
                "k_original": module.gate_up_k,
                "k_gemm": module.gate_up_k_gemm,
                "tile_size": module.TILE_SIZE,
            }

            # Handle down_proj weights (MXFP4 format)
            target_blocks_down = f"{name}.down_proj_blocks"
            source_blocks_down = f"{name}.down_proj_blocks"
            fusion_map[target_blocks_down] = {
                "sources": [source_blocks_down],
                "op": "load_mxfp4_blocks_padded",
                "k_original": module.down_k,
                "k_gemm": module.down_k_gemm,
            }

            target_scales_down = f"{name}.down_proj_scales"
            source_scales_down = f"{name}.down_proj_scales"
            fusion_map[target_scales_down] = {
                "sources": [source_scales_down],
                "op": "load_mxfp4_scales_swizzled",
                "n": module.hidden_size,
                "n_padded": module.down_n_padded,
                "k_original": module.down_k,
                "k_gemm": module.down_k_gemm,
                "tile_size": module.TILE_SIZE,
            }

    return fusion_map


class GptOssAttention(nn.Module):
    """GPT OSS attention module with attention sink."""

    def __init__(self, config: GptOssArch, layer_idx: int, rope: GptOssRotaryEmbedding):
        """Initialize the GPT OSS attention module."""
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = config.head_size
        self.num_attention_heads = config.num_query_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads
        # Apply sliding window to even layers and full attention to odd layers
        # This follows the GPT-OSS alternating attention pattern
        self.sliding_window = config.sliding_window if layer_idx % 2 == 0 else 0

        # Define the output sizes for Q, K, and V for clarity
        self.q_size = config.num_query_heads * config.head_size
        self.k_size = config.num_key_value_heads * config.head_size
        self.v_size = config.num_key_value_heads * config.head_size

        # Sink tokens parameter
        self.sinks = nn.Parameter(
            torch.empty(
                config.num_query_heads,
                device=torch.device(config.device),
                dtype=config.dtype,
            )
        )

        qkv_dim = config.head_size * (
            config.num_query_heads + 2 * config.num_key_value_heads
        )
        self.qkv_proj = nn.Linear(
            config.hidden_size,
            qkv_dim,
            device=torch.device(config.device),
            dtype=config.dtype,
        )

        self.o_proj = nn.Linear(
            config.head_size * config.num_query_heads,
            config.hidden_size,
            device=torch.device(config.device),
            dtype=config.dtype,
        )

        self.scaling = self.head_dim**-0.5

        self.rope = rope

    def _ceil_div(self, a: int, b: int) -> int:
        return -(-a // b)

    def _attend_one_page(
        self,
        query: torch.Tensor,
        paged_keys: torch.Tensor,
        paged_mask: torch.Tensor,
        paged_values: torch.Tensor,
        sum_exp: torch.Tensor,
        sum_val: torch.Tensor,
        max_score: torch.Tensor,
    ):
        page_attn_scores = einsum(query, paged_keys, "q h d, s h d -> h q s")
        page_attn_scores = (page_attn_scores + paged_mask).to(torch.float32)
        page_max_score = torch.max(page_attn_scores, dim=-1, keepdim=False).values

        # Convert -inf elements to 0.0 in page_max_score
        page_max_score = torch.where(
            torch.isinf(page_max_score) & (page_max_score < 0),
            torch.tensor(0.0, dtype=page_max_score.dtype, device=page_max_score.device),
            page_max_score,
        )

        page_attn_scores = torch.exp(page_attn_scores - page_max_score.unsqueeze(-1))

        page_sum_exp = torch.sum(page_attn_scores, dim=-1, keepdim=False)
        page_sum_val = einsum(
            page_attn_scores, paged_values.to(torch.float32), "h q s, s h d -> h q d"
        )

        new_max_score = torch.max(max_score, page_max_score)
        alpha = torch.exp(max_score - new_max_score)
        beta = torch.exp(page_max_score - new_max_score)

        sum_exp = sum_exp * alpha + page_sum_exp * beta
        sum_val = sum_val * alpha.unsqueeze(-1) + page_sum_val * beta.unsqueeze(-1)
        max_score = new_max_score

        return sum_val, sum_exp, max_score

    def _paged_attention(
        self,
        queries: torch.Tensor,
        qo_indptr: torch.IntTensor,
        kv_page_indptr: torch.IntTensor,
        kv_last_page_lens: torch.IntTensor,
        kv_page_indices: torch.IntTensor,
        attention_mask: torch.Tensor,
        kv_cache_at_layer: torch.Tensor,
    ):
        output_embeds = torch.empty(
            queries.shape[0],
            self.config.hidden_size,
            dtype=queries.dtype,
            device=queries.device,
        )
        kv_page_size = kv_cache_at_layer.shape[2]
        mask_offset = 0
        batch_num = len(qo_indptr) - 1

        for batch_idx in range(batch_num):
            q_start = qo_indptr[batch_idx]
            q_end = qo_indptr[batch_idx + 1]
            query_len = int(q_end - q_start)

            kv_page_start = int(kv_page_indptr[batch_idx])
            kv_page_end = int(kv_page_indptr[batch_idx + 1])
            kv_last_page_len = int(kv_last_page_lens[batch_idx])

            seq_len = int(
                (kv_page_end - kv_page_start - 1) * kv_page_size + kv_last_page_len
            )

            mask_len = seq_len * query_len
            mask = attention_mask[mask_offset : mask_offset + mask_len].view(
                query_len, seq_len
            )
            mask_offset += mask_len

            # If this attention layer uses a sliding window, we keep only the last
            # few pages of the KV cache and the corresponding mask.
            if self.sliding_window != 0:
                attn_page_cnt = 1 + self._ceil_div(
                    self.sliding_window - kv_last_page_len, kv_page_size
                )
                kv_page_start = max(kv_page_start, kv_page_end - attn_page_cnt)

                seq_len = int(
                    (kv_page_end - kv_page_start - 1) * kv_page_size + kv_last_page_len
                )

                mask = mask[:, -seq_len:]

            query = queries[q_start:q_end, :] * self.scaling

            sum_exp = torch.zeros(
                self.num_attention_heads,
                query_len,
                device=query.device,
                dtype=torch.float32,
            )
            sum_val = torch.zeros(
                self.num_attention_heads,
                query_len,
                self.head_dim,
                device=query.device,
                dtype=torch.float32,
            )
            max_score = torch.zeros(
                self.num_attention_heads,
                query_len,
                device=query.device,
                dtype=torch.float32,
            )

            # Attend to all but the last page, processing 32 pages at a time
            for page_cnts, kv_page_idx_idxs in chunked_enumerate(
                range(kv_page_start, kv_page_end - 1), 32
            ):
                chunk_kv_page_indices = kv_page_indices[kv_page_idx_idxs]

                # Gather keys and values for all pages in the chunk at once
                # Shape: [chunk_size, page_size, num_kv_heads, head_dim]
                chunk_keys = kv_cache_at_layer[chunk_kv_page_indices, 0]
                chunk_values = kv_cache_at_layer[chunk_kv_page_indices, 1]

                # Reshape to concatenate pages as one page:
                # [chunk_size * page_size, num_kv_heads, head_dim]
                paged_keys = chunk_keys.view(
                    -1, chunk_keys.shape[-2], chunk_keys.shape[-1]
                )
                paged_values = chunk_values.view(
                    -1, chunk_values.shape[-2], chunk_values.shape[-1]
                )

                paged_keys = torch.repeat_interleave(
                    paged_keys, self.num_key_value_groups, dim=1
                )
                paged_values = torch.repeat_interleave(
                    paged_values, self.num_key_value_groups, dim=1
                )

                chunk_size = len(page_cnts)
                mask_start = page_cnts[0] * kv_page_size
                mask_end = mask_start + chunk_size * kv_page_size
                paged_mask = mask[:, mask_start:mask_end].unsqueeze(0)

                sum_val, sum_exp, max_score = self._attend_one_page(
                    query,
                    paged_keys,
                    paged_mask,
                    paged_values,
                    sum_exp,
                    sum_val,
                    max_score,
                )

            # Attend to the last page
            page_cnt = kv_page_end - kv_page_start - 1
            kv_page_idx_idx = kv_page_end - 1

            kv_page_idx = kv_page_indices[kv_page_idx_idx]
            paged_keys = kv_cache_at_layer[kv_page_idx, 0][:kv_last_page_len]
            paged_values = kv_cache_at_layer[kv_page_idx, 1][:kv_last_page_len]

            paged_keys = torch.repeat_interleave(
                paged_keys, self.num_key_value_groups, dim=1
            )
            paged_values = torch.repeat_interleave(
                paged_values, self.num_key_value_groups, dim=1
            )

            paged_mask_offset = page_cnt * kv_page_size
            paged_mask = mask[:, paged_mask_offset : paged_mask_offset + kv_page_size][
                ..., :kv_last_page_len
            ]
            paged_mask = paged_mask.unsqueeze(0)

            sum_val, sum_exp, max_score = self._attend_one_page(
                query, paged_keys, paged_mask, paged_values, sum_exp, sum_val, max_score
            )

            adjusted_sinks = self.sinks.unsqueeze(-1) - max_score
            adjusted_sinks = torch.exp(adjusted_sinks)
            sum_exp += adjusted_sinks

            attn_output = sum_val / sum_exp.unsqueeze(-1)
            attn_output = rearrange(attn_output, "h q d -> q (h d)")

            attn_output = self.o_proj(attn_output.to(queries.dtype))

            output_embeds[q_start:q_end, :] = attn_output

        return output_embeds

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.IntTensor,
        qo_indptr: torch.IntTensor,
        kv_cache_at_layer: torch.Tensor,
        kv_page_indices: torch.IntTensor,
        kv_page_indptr: torch.IntTensor,
        kv_last_page_lens: torch.IntTensor,
        batch_indices: torch.IntTensor,
        batch_positions: torch.IntTensor,
        attention_mask: torch.Tensor,
        adapter_subpass: AdapterSubpass | None,
    ) -> torch.Tensor:
        """Forward pass through the attention module."""
        n, _ = hidden_states.size()

        qkv_states = self.qkv_proj(hidden_states)

        query_states, key_states, value_states = torch.split(
            qkv_states, [self.q_size, self.k_size, self.v_size], dim=-1
        )

        # apply adapters if provided
        if adapter_subpass is not None:
            adapter_subpass.execute(
                self.layer_idx,
                hidden_states,
                q_state=query_states,
                k_state=key_states,
                v_state=value_states,
            )

        # Reshape for multi-head attention
        query_states = query_states.view(n, self.num_attention_heads, self.head_dim)
        key_states = key_states.view(n, self.num_key_value_heads, self.head_dim)
        value_states = value_states.view(n, self.num_key_value_heads, self.head_dim)

        # Apply rotary embedding
        query_states, key_states = self.rope(query_states, key_states, position_ids)

        # Store current KV states in FlashInfer cache for future use
        ops.append_paged_kv_cache(
            append_key=key_states,
            append_value=value_states,
            batch_indices=batch_indices,
            positions=batch_positions,
            paged_kv_cache=kv_cache_at_layer[self.layer_idx],
            kv_indices=kv_page_indices,
            kv_indptr=kv_page_indptr,
            kv_last_page_len=kv_last_page_lens,
            kv_layout="NHD",
        )

        new_attn_output = self._paged_attention(
            query_states,
            qo_indptr,
            kv_page_indptr,
            kv_last_page_lens,
            kv_page_indices,
            attention_mask,
            kv_cache_at_layer[self.layer_idx],
        )

        return new_attn_output


class GptOssRouter(nn.Module):
    """GPT OSS Router for selecting top-k experts."""

    def __init__(self, config: GptOssArch):
        """Initialize the GPT OSS Router."""
        super().__init__()
        self.experts_per_token = config.experts_per_token
        self.num_experts = config.num_experts
        self.hidden_size = config.hidden_size

        self.weight = nn.Parameter(
            torch.empty(
                config.num_experts,
                config.hidden_size,
                device=torch.device(config.device),
                dtype=config.dtype,
            )
        )
        self.bias = nn.Parameter(
            torch.empty(
                config.num_experts,
                device=torch.device(config.device),
                dtype=config.dtype,
            )
        )

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the router."""
        hidden_states = hidden_states.reshape(-1, self.hidden_size)
        router_logits = torch.nn.functional.linear(  # pylint: disable=not-callable
            hidden_states, self.weight, self.bias
        )

        router_top_value, router_indices = torch.topk(
            router_logits, self.experts_per_token, dim=-1, sorted=True
        )

        router_top_value = torch.nn.functional.softmax(router_top_value, dim=1)
        router_scores = torch.zeros_like(router_logits).scatter_(
            1, router_indices, router_top_value
        )
        return router_scores, router_indices


class GptOssExperts(nn.Module):
    """GPT OSS Experts layer with native MXFP4 support via FlashInfer.
    
    This implementation stores weights in MXFP4 format and uses FlashInfer's
    group_gemm_mxfp4_nt_groupwise for efficient computation on Blackwell GPUs.
    
    Weight storage:
    - gate_up_proj_blocks: (num_experts, N, K//2) uint8 - packed FP4 values
    - gate_up_proj_scales: (num_experts, N, K//32) uint8 - UE8M0 scales (pre-swizzled)
    - down_proj_blocks: (num_experts, N, K//2) uint8
    - down_proj_scales: (num_experts, N, K//32) uint8 (pre-swizzled)
    - Biases remain in bfloat16
    
    Runtime:
    - Activations are quantized to MXFP8 using mxfp8_quantize
    - group_gemm_mxfp4_nt_groupwise computes the GEMM
    """

    # Constants for MXFP4 quantization
    TILE_SIZE = 32
    ALIGNMENT_N = 8
    ALIGNMENT_K = 128
    ALIGNMENT_M_SF = 128

    def __init__(self, config: GptOssArch):
        """Initialize the GPT OSS Experts layer with MXFP4 weights."""
        super().__init__()
        self.config = config
        self.num_experts = config.num_experts
        self.experts_per_token = config.experts_per_token
        self.swiglu_limit = config.swiglu_limit
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.device = torch.device(config.device)
        self.dtype = config.dtype

        # Compute dimensions for gate_up projection
        # K = hidden_size, N = intermediate_size * 2
        self.gate_up_k = config.hidden_size
        self.gate_up_k_groups = config.hidden_size // self.TILE_SIZE  # Actual (non-padded)
        # k_aligned is K aligned to TILE_SIZE (32) - what mxfp8_quantize produces
        self.gate_up_k_aligned = ((config.hidden_size + self.TILE_SIZE - 1) // self.TILE_SIZE) * self.TILE_SIZE
        self.gate_up_k_groups_aligned = self.gate_up_k_aligned // self.TILE_SIZE
        # k_gemm is K aligned to ALIGNMENT_K (128) - required by FlashInfer GEMM
        self.gate_up_k_gemm = ((config.hidden_size + self.ALIGNMENT_K - 1) // self.ALIGNMENT_K) * self.ALIGNMENT_K
        self.gate_up_k_groups_gemm = self.gate_up_k_gemm // self.TILE_SIZE
        
        gate_up_n = config.intermediate_size * 2
        self.gate_up_n_padded = ((gate_up_n + self.ALIGNMENT_N - 1) // self.ALIGNMENT_N) * self.ALIGNMENT_N

        # Compute dimensions for down projection
        # K = intermediate_size, N = hidden_size
        self.down_k = config.intermediate_size
        self.down_k_groups = config.intermediate_size // self.TILE_SIZE  # Actual (non-padded)
        # k_aligned is K aligned to TILE_SIZE (32) - what mxfp8_quantize produces
        self.down_k_aligned = ((config.intermediate_size + self.TILE_SIZE - 1) // self.TILE_SIZE) * self.TILE_SIZE
        self.down_k_groups_aligned = self.down_k_aligned // self.TILE_SIZE
        self.down_k_gemm = ((config.intermediate_size + self.ALIGNMENT_K - 1) // self.ALIGNMENT_K) * self.ALIGNMENT_K
        self.down_k_groups_gemm = self.down_k_gemm // self.TILE_SIZE
        
        down_n = config.hidden_size
        self.down_n_padded = ((down_n + self.ALIGNMENT_N - 1) // self.ALIGNMENT_N) * self.ALIGNMENT_N
        
        # Legacy aliases for compatibility
        self.gate_up_k_padded = self.gate_up_k_gemm
        self.down_k_padded = self.down_k_gemm
        self.k_padded = self.gate_up_k_gemm
        self.k_groups = self.gate_up_k_groups_gemm

        # MXFP4 weight storage (packed FP4 blocks + UE8M0 scales)
        # Weights are PRE-PADDED to k_gemm alignment at load time for efficiency
        # gate_up: weights are (num_experts, intermediate*2, hidden)
        # Stored as (num_experts, intermediate*2, k_gemm//2) packed - PRE-PADDED
        self.register_buffer(
            "gate_up_proj_blocks",
            torch.zeros(
                (config.num_experts, gate_up_n, self.gate_up_k_gemm // 2),
                dtype=torch.uint8,
                device=self.device,
            ),
        )
        # Scales PRE-SWIZZLED at load time for efficiency
        # Note: _pad_scale_factors pads N to max(n_padded, k_gemm) for alignment
        gate_up_scale_n = max(self.gate_up_n_padded, self.gate_up_k_gemm)
        self.register_buffer(
            "gate_up_proj_scales",
            torch.zeros(
                (config.num_experts, gate_up_scale_n, self.gate_up_k_groups_gemm),
                dtype=torch.uint8,
                device=self.device,
            ),
        )

        # down: weights are (num_experts, hidden, intermediate)
        # Stored as (num_experts, hidden, k_gemm//2) packed - PRE-PADDED
        self.register_buffer(
            "down_proj_blocks",
            torch.zeros(
                (config.num_experts, config.hidden_size, self.down_k_gemm // 2),
                dtype=torch.uint8,
                device=self.device,
            ),
        )
        # Scales PRE-SWIZZLED at load time
        # Note: _pad_scale_factors pads N to max(n_padded, k_gemm) for alignment
        down_scale_n = max(self.down_n_padded, self.down_k_gemm)
        self.register_buffer(
            "down_proj_scales",
            torch.zeros(
                (config.num_experts, down_scale_n, self.down_k_groups_gemm),
                dtype=torch.uint8,
                device=self.device,
            ),
        )

        # Biases remain in bfloat16
        self.gate_up_proj_bias = nn.Parameter(
            torch.empty(
                (config.num_experts, config.intermediate_size * 2),
                device=self.device,
                dtype=self.dtype,
            )
        )
        self.down_proj_bias = nn.Parameter(
            torch.empty(
                (config.num_experts, config.hidden_size),
                device=self.device,
                dtype=self.dtype,
            )
        )
        
    def _prepare_activation_scales(
        self, a_scale: torch.Tensor, num_groups: int, m_per_group: int, k_padded: int
    ) -> torch.Tensor:
        """Prepare activation scales: swizzle and add per-group padding.
        
        Args:
            a_scale: Scale tensor from mxfp8_quantize, shape (cum_m, k_groups)
            num_groups: Number of expert groups
            m_per_group: Tokens per group (must be uniform for this implementation)
            k_padded: Padded K dimension
        
        Returns:
            Swizzled and padded scale tensor ready for group_gemm
        """
        k_groups = k_padded // self.TILE_SIZE
        cum_m = num_groups * m_per_group

        # Reshape for swizzling: (num_groups, m_per_group, k_groups)
        a_scale_3d = a_scale.reshape(num_groups, m_per_group, k_groups)

        # Swizzle
        a_scale_swizzled = _swizzle_blockscale(
            a_scale_3d, num_groups, m_per_group, k_padded, self.TILE_SIZE
        ).flatten(0, 1)

        # Apply per-group padding to multiples of ALIGNMENT_M_SF
        group_arange = torch.arange(0, num_groups + 1, dtype=torch.int32, device=self.device)
        m_indptr = group_arange * m_per_group
        m_indptr_padded = (
            (m_indptr + group_arange * (self.ALIGNMENT_M_SF - 1))
            // self.ALIGNMENT_M_SF
            * self.ALIGNMENT_M_SF
        )
        m_sf = m_indptr_padded[1:] - m_indptr_padded[:-1]

        # Pad each group's scales
        a_scale_chunks = a_scale_swizzled.chunk(num_groups, dim=0)
        a_scale_padded = [
            torch.cat([
                chunk,
                torch.zeros(
                    m_sf[i].item() - chunk.shape[0],
                    *chunk.shape[1:],
                    dtype=chunk.dtype,
                    device=chunk.device,
                ),
            ])
            for i, chunk in enumerate(a_scale_chunks)
        ]
        return torch.cat(a_scale_padded)

    def forward(self, t: torch.Tensor, expert_indices: torch.Tensor) -> torch.Tensor:
        """Forward pass through the experts using native MXFP4 computation.
        
        Args:
            t: Input tensor of shape (batch_size, hidden_size)
            expert_indices: Expert assignments of shape (batch_size, experts_per_token)
        
        Returns:
            Output tensor of shape (batch_size, experts_per_token, hidden_size)
        """
        batch_size = t.shape[0]
        experts_per_token = expert_indices.shape[1]

        # ================================================================
        # Reorganize by expert for efficient batched computation
        # ================================================================
        # Flatten expert assignments: (batch * experts_per_token,)
        flat_expert_ids = expert_indices.flatten()  # (B * E,)

        # Create token indices for each expert call (token_idx = flat_idx // experts_per_token)
        total_elements = batch_size * experts_per_token
        token_indices = torch.arange(total_elements, device=self.device) // experts_per_token

        # Sort by expert ID to group tokens going to the same expert
        sorted_indices = torch.argsort(flat_expert_ids, stable=True)
        sorted_expert_ids = flat_expert_ids[sorted_indices]
        sorted_token_indices = token_indices[sorted_indices]

        # Find unique experts and their counts
        unique_experts, inverse_indices, expert_counts = torch.unique(
            sorted_expert_ids, return_inverse=True, return_counts=True
        )
        num_active_experts = unique_experts.shape[0]

        # Build m_indptr for group_gemm
        m_indptr = torch.zeros(num_active_experts + 1, dtype=torch.int32, device=self.device)
        m_indptr[1:] = torch.cumsum(expert_counts, dim=0).to(torch.int32)
        cum_m = m_indptr[-1].item()

        # Gather activations in sorted order: (cum_m, hidden_size)
        sorted_activations = t[sorted_token_indices]

        # ================================================================
        # Gate + Up projection: (cum_m, hidden) -> (cum_m, intermediate*2)
        # ================================================================
        # Quantize activations to MXFP8
        # mxfp8_quantize pads K to multiples of TILE_SIZE (32)
        # The K dimension must match the weight's K (gate_up_k = hidden_size)
        a_fp8, a_scale_flat = mxfp8_quantize(
            sorted_activations.to(torch.bfloat16),
            is_sf_swizzled_layout=False,
            alignment=self.TILE_SIZE,
        )

        # mxfp8_quantize returns:
        # - a_fp8: (cum_m, k_aligned) where k_aligned = ceil(K / 32) * 32
        # - a_scale_flat: (cum_m * k_groups,) where k_groups = k_aligned / 32
        # For hidden_size=2880: k_aligned=2880, k_groups=90
        a_scale = a_scale_flat.reshape(cum_m, self.gate_up_k_groups_aligned)

        # Prepare scales with swizzling and padding
        # For variable group sizes, we need a different approach
        # For now, process groups that have the same size together
        # This is a simplified implementation - full version would handle variable sizes

        # Gather weights for active experts (pre-padded and pre-swizzled at load time)
        active_gate_up_blocks = self.gate_up_proj_blocks[unique_experts]  # (num_active, N, k_gemm//2)
        active_gate_up_scales = self.gate_up_proj_scales[unique_experts]  # (num_active, N_pad, k_groups_gemm) - swizzled

        # Compute gate_up projection using group_gemm
        # Weights are pre-padded to k_gemm and scales are pre-swizzled
        gate_up_out = self._group_gemm_forward(
            a_fp8=a_fp8,
            a_scale=a_scale,
            b_blocks=active_gate_up_blocks,
            b_scales_swizzled=active_gate_up_scales,
            m_indptr=m_indptr,
            n=self.intermediate_size * 2,
            k_gemm=self.gate_up_k_gemm,
        )

        # Add bias
        active_gate_up_bias = self.gate_up_proj_bias[unique_experts]  # (num_active, N)
        # Expand bias to match tokens: use inverse_indices to map back
        bias_expanded = active_gate_up_bias[inverse_indices]  # (cum_m, N)
        gate_up_out = gate_up_out + bias_expanded

        # ================================================================
        # SwiGLU activation
        # ================================================================
        x_glu = gate_up_out[..., ::2]
        x_linear = gate_up_out[..., 1::2]

        x_glu = x_glu.clamp(min=None, max=self.swiglu_limit)
        x_linear = x_linear.clamp(min=-self.swiglu_limit, max=self.swiglu_limit)
        out_glu = x_glu * torch.sigmoid(1.702 * x_glu)

        hidden_states = out_glu * (x_linear + 1)  # (cum_m, intermediate)

        # ================================================================
        # Down projection: (cum_m, intermediate) -> (cum_m, hidden)
        # ================================================================
        # Quantize hidden states to MXFP8
        h_fp8, h_scale_flat = mxfp8_quantize(
            hidden_states.to(torch.bfloat16),
            is_sf_swizzled_layout=False,
            alignment=self.TILE_SIZE,
        )

        # mxfp8_quantize returns:
        # - h_fp8: (cum_m, k_aligned) where k_aligned = ceil(K / 32) * 32
        # - h_scale_flat: (cum_m * k_groups,)
        # For intermediate_size=2880: k_aligned=2880, k_groups=90
        h_scale = h_scale_flat.reshape(cum_m, self.down_k_groups_aligned)

        # Gather weights for active experts (pre-padded and pre-swizzled at load time)
        active_down_blocks = self.down_proj_blocks[unique_experts]  # (num_active, N, k_gemm//2)
        active_down_scales = self.down_proj_scales[unique_experts]  # (num_active, N_pad, k_groups_gemm) - swizzled

        # Compute down projection
        # Weights are pre-padded to k_gemm and scales are pre-swizzled
        down_out = self._group_gemm_forward(
            a_fp8=h_fp8,
            a_scale=h_scale,
            b_blocks=active_down_blocks,
            b_scales_swizzled=active_down_scales,
            m_indptr=m_indptr,
            n=self.hidden_size,
            k_gemm=self.down_k_gemm,
        )

        # Add bias
        active_down_bias = self.down_proj_bias[unique_experts]
        bias_expanded = active_down_bias[inverse_indices]
        down_out = down_out + bias_expanded  # (cum_m, hidden)

        # ================================================================
        # Scatter results back to original order
        # ================================================================
        # Create output tensor
        output = torch.zeros(
            batch_size * experts_per_token,
            self.hidden_size,
            dtype=self.dtype,
            device=self.device,
        )
        output[sorted_indices] = down_out

        # Reshape to (batch, experts_per_token, hidden)
        output = output.reshape(batch_size, experts_per_token, self.hidden_size)

        return output

    def _group_gemm_forward(
        self,
        a_fp8: torch.Tensor,
        a_scale: torch.Tensor,
        b_blocks: torch.Tensor,
        b_scales_swizzled: torch.Tensor,
        m_indptr: torch.Tensor,
        n: int,
        k_gemm: int,
    ) -> torch.Tensor:
        """Execute group GEMM using FlashInfer's native MXFP4 kernel.
        
        Args:
            a_fp8: FP8 activations, shape (cum_m, k_aligned) - will be padded to k_gemm
            a_scale: Activation scales, shape (cum_m, k_groups) - will be padded
            b_blocks: FP4 weights PRE-PADDED, shape (num_groups, n, k_gemm//2)
            b_scales_swizzled: Weight scales PRE-SWIZZLED, shape (num_groups, n_padded, k_groups_gemm)
            m_indptr: Group boundaries, shape (num_groups + 1,)
            n: Output dimension (before padding)
            k_gemm: K aligned to ALIGNMENT_K (128) - for activation padding
        
        Returns:
            Output tensor, shape (cum_m, n)
        """
        num_groups = b_blocks.shape[0]
        cum_m = a_fp8.shape[0]
        k_groups_gemm = k_gemm // self.TILE_SIZE
        
        # Pad activations to k_gemm if needed (weights are pre-padded at load time)
        k_aligned = a_fp8.shape[1]
        if k_aligned < k_gemm:
            a_fp8_padded = torch.zeros(
                (cum_m, k_gemm), dtype=a_fp8.dtype, device=a_fp8.device
            )
            a_fp8_padded[:, :k_aligned] = a_fp8
            a_fp8 = a_fp8_padded
            
            # Pad activation scales
            k_groups = k_aligned // self.TILE_SIZE
            a_scale_padded = torch.full(
                (cum_m, k_groups_gemm), 127, dtype=torch.uint8, device=a_scale.device
            )
            a_scale_padded[:, :k_groups] = a_scale
            a_scale = a_scale_padded
        
        k_groups = k_groups_gemm
        
        # NOTE: Weight blocks and scales are PRE-PADDED and PRE-SWIZZLED at load time
        # No runtime padding/swizzling needed for weights

        # Handle the case where different groups have different sizes
        # Also need to ensure groups are padded to multiple of 4 for swizzling
        group_sizes = m_indptr[1:] - m_indptr[:-1]
        
        # Use aminmax() to get both min and max in single kernel call
        min_val, max_val = group_sizes.aminmax()
        min_group_size = min_val.item()
        max_group_size = max_val.item()

        # Ensure max_group_size is multiple of 4
        max_group_size_padded = ((max_group_size + 3) // 4) * 4

        # Prepare activation scales with swizzling and padding
        # Need to pad when:
        # 1. Group sizes vary (not all same)
        # 2. max_group_size needs alignment padding to multiple of 4
        needs_padding = (
            min_group_size != max_group_size or 
            max_group_size != max_group_size_padded
        )
        
        if needs_padding:
            # Pad activations and scales to uniform group sizes
            a_fp8_padded = torch.zeros(
                num_groups * max_group_size_padded, k_gemm,
                dtype=a_fp8.dtype, device=a_fp8.device
            )
            a_scale_padded = torch.full(
                (num_groups * max_group_size_padded, k_groups),
                127,  # Neutral scale
                dtype=torch.uint8, device=a_scale.device
            )

            # Convert to Python list once to avoid repeated .item() calls
            m_indptr_list = m_indptr.tolist()
            for i in range(num_groups):
                start = m_indptr_list[i]
                end = m_indptr_list[i + 1]
                size = end - start
                if size > 0:
                    padded_start = i * max_group_size_padded
                    a_fp8_padded[padded_start:padded_start + size] = a_fp8[start:end]
                    a_scale_padded[padded_start:padded_start + size] = a_scale[start:end]

            a_fp8 = a_fp8_padded
            a_scale = a_scale_padded
            
            # Update m_indptr for padded layout
            m_indptr = torch.arange(
                0, (num_groups + 1) * max_group_size_padded, max_group_size_padded,
                dtype=torch.int32, device=self.device
            )

        # Swizzle activation scales
        a_scale_3d = a_scale.reshape(num_groups, max_group_size_padded, k_groups)
        a_scale_swizzled = _swizzle_blockscale(
            a_scale_3d, num_groups, max_group_size_padded, k_gemm, self.TILE_SIZE
        ).flatten(0, 1)

        # Apply per-group ALIGNMENT_M_SF padding for a_scale
        # This formula is reverse-engineered from FlashInfer's internal scale layout expectations.
        # Source: https://github.com/flashinfer-ai/flashinfer/blob/main/tests/gemm/test_groupwise_scaled_gemm_mxfp4.py
        # FlashInfer expects: m_indptr_padded[i] = ceil((m_indptr[i] + i*(ALIGNMENT-1)) / ALIGNMENT) * ALIGNMENT
        # The "+ i * (ALIGNMENT - 1)" term ensures proper cumulative alignment across groups.
        group_arange = torch.arange(0, num_groups + 1, dtype=torch.int32, device=self.device)
        m_indptr_for_sf = group_arange * max_group_size_padded
        m_indptr_padded = (
            (m_indptr_for_sf + group_arange * (self.ALIGNMENT_M_SF - 1))
            // self.ALIGNMENT_M_SF
            * self.ALIGNMENT_M_SF
        )
        m_sf = m_indptr_padded[1:] - m_indptr_padded[:-1]

        # Build padded scale tensor
        # Each chunk from swizzle has shape (padded_m, k_sf_cols) where padded_m = ceil(max_group_size_padded/128)*128
        # m_sf[i] should equal padded_m for all groups (by design of the padding formula)
        m_sf_list = m_sf.tolist()
        a_scale_chunks = a_scale_swizzled.chunk(num_groups, dim=0)
        
        # Check if any padding is needed (usually not, since swizzle already pads to 128)
        chunk_size = a_scale_chunks[0].shape[0]
        needs_extra_padding = any(m_sf_list[i] != chunk_size for i in range(num_groups))
        
        if needs_extra_padding:
            # Rare case: need additional padding per chunk
            a_scale_final = torch.cat([
                torch.cat([
                    chunk,
                    torch.zeros(
                        m_sf_list[i] - chunk.shape[0],
                        *chunk.shape[1:],
                        dtype=chunk.dtype,
                        device=chunk.device,
                    ),
                ])
                for i, chunk in enumerate(a_scale_chunks)
            ])
        else:
            # Common case: chunks are already correctly sized, just concatenate
            a_scale_final = a_scale_swizzled

        # Call FlashInfer group GEMM
        output_padded = group_gemm_mxfp4_nt_groupwise(
            a_fp8,
            b_blocks,
            a_scale_final,
            b_scales_swizzled,
            m_indptr,
            mma_sm=1,
            tile_m=128,
            tile_n=128,
            tile_k=128,
            swap_ab=True,
            out_dtype=self.dtype,
        )

        # Extract valid outputs (remove padding)
        if output_padded.shape[0] != cum_m:
            orig_m_indptr = torch.zeros(num_groups + 1, dtype=torch.int32, device=self.device)
            orig_m_indptr[1:] = torch.cumsum(group_sizes, dim=0).to(torch.int32)
            
            # Convert to Python lists once to avoid repeated .item() calls
            group_sizes_list = group_sizes.tolist()
            orig_m_indptr_list = orig_m_indptr.tolist()
            
            output = torch.zeros(cum_m, n, dtype=self.dtype, device=self.device)
            for i in range(num_groups):
                size = group_sizes_list[i]
                if size > 0:
                    src_start = i * max_group_size_padded
                    dst_start = orig_m_indptr_list[i]
                    output[dst_start:dst_start + size] = output_padded[src_start:src_start + size, :n]
            return output
        else:
            return output_padded[:, :n]


class GptOssMlp(nn.Module):
    """GPT OSS MLP layer with Mixture of Experts."""

    def __init__(self, config: GptOssArch):
        """Initialize the GPT OSS MLP layer."""
        super().__init__()
        self.config = config
        self.router = GptOssRouter(config)
        self.experts = GptOssExperts(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the MLP layer."""
        # Router determines expert selection and weights
        router_scores, router_indices = self.router(x)

        # Extract the weights for selected experts
        expert_weights = torch.gather(router_scores, 1, router_indices)

        # Forward through experts
        t = self.experts(x, router_indices)

        # Weighted sum of experts
        t = torch.einsum("bec,be->bc", t, expert_weights)

        return t


class GptOssDecoderLayer(nn.Module):
    """GPT OSS decoder layer."""

    def __init__(self, config: GptOssArch, layer_idx: int):
        """Initialize the GPT OSS decoder layer."""
        super().__init__()
        self.layer_idx = layer_idx
        self.input_layernorm = GptOssRMSNorm(config.hidden_size, device=config.device)
        self.rope = GptOssRotaryEmbedding(
            config.head_size,
            int(config.rope_theta),
            torch.float32,
            initial_context_length=config.initial_context_length,
            scaling_factor=config.rope_scaling_factor,
            ntk_alpha=config.rope_ntk_alpha,
            ntk_beta=config.rope_ntk_beta,
            device=torch.device(config.device),
            max_position_id=131072,
        )
        self.self_attn = GptOssAttention(config, layer_idx, self.rope)
        self.mlp = GptOssMlp(config)
        self.post_attention_layernorm = GptOssRMSNorm(
            config.hidden_size, device=config.device
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        qo_indptr: torch.Tensor,
        kv_cache_at_layer: torch.Tensor,
        kv_page_indices: torch.Tensor,
        kv_page_indptr: torch.Tensor,
        kv_last_page_lens: torch.Tensor,
        batch_indices: torch.Tensor,
        batch_positions: torch.Tensor,
        full_mask: torch.Tensor,
        window_mask: torch.Tensor,
        adapter_subpass: AdapterSubpass | None,
    ) -> torch.Tensor:
        """Forward pass through the decoder layer."""
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            position_ids=position_ids,
            qo_indptr=qo_indptr,
            kv_cache_at_layer=kv_cache_at_layer,
            kv_page_indices=kv_page_indices,
            kv_page_indptr=kv_page_indptr,
            kv_last_page_lens=kv_last_page_lens,
            batch_indices=batch_indices,
            batch_positions=batch_positions,
            attention_mask=window_mask if self.layer_idx % 2 == 0 else full_mask,
            adapter_subpass=adapter_subpass,
        )

        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        hidden_states = self.mlp(hidden_states)

        hidden_states = residual + hidden_states

        return hidden_states


def _dequantize_mxfp4_to_bf16(
    blocks: torch.Tensor,
    scales: torch.Tensor,
    fp4_values: tuple[float, ...],
    device: str,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """
    Dequantize MXFP4 format tensors (blocks and scales) to bfloat16.

    This is a copy of the dequantization logic from model_loader.py for local testing.

    Args:
        blocks: The packed FP4 values tensor (uint8), shape (..., group_count, block_size)
        scales: The block scales tensor (uint8), shape (..., group_count)
        fp4_values: Lookup table for FP4 values
        device: Target device
        dtype: Target dtype (default: torch.bfloat16)

    Returns:
        Dequantized tensor in the target dtype
    """
    scales = scales.to(torch.int32) - 127

    assert (
        blocks.shape[:-1] == scales.shape
    ), f"{blocks.shape=} does not match {scales.shape=}"

    lut = torch.tensor(fp4_values, dtype=dtype, device=device)

    *prefix_shape, g, b = blocks.shape
    rows_total = math.prod(prefix_shape) * g

    blocks = blocks.reshape(rows_total, b).to(device)
    scales = scales.reshape(rows_total, 1).to(device)

    # Extract low and high 4-bit indices
    # FlashInfer FP4 nibble packing order (reverse-engineered by testing):
    # Packing: (even_index << 4) | odd_index
    # - High nibble (>> 4) contains even indices (0, 2, 4, ...)
    # - Low nibble (& 0x0F) contains odd indices (1, 3, 5, ...)
    idx_hi = (blocks >> 4).to(torch.long)    # Even indices (high nibble)
    idx_lo = (blocks & 0x0F).to(torch.long)  # Odd indices (low nibble)

    # Create output tensor and populate
    out = torch.empty(rows_total, b * 2, dtype=dtype, device=device)
    out[:, 0::2] = lut[idx_hi]  # High nibble values at even indices
    out[:, 1::2] = lut[idx_lo]  # Low nibble values at odd indices

    torch.ldexp(out, scales, out=out)

    return out.reshape(*prefix_shape, g, b * 2).view(*prefix_shape, g * b * 2)


def _swizzle_blockscale(unswizzled_sf: torch.Tensor, b: int, m: int, n: int, sf_vec_size: int = 32) -> torch.Tensor:
    """
    Swizzle block scale tensor for MXFP4/MXFP8 format.
    
    This transforms the scale factors into the memory layout expected by FlashInfer's
    group_gemm_mxfp4_nt_groupwise kernel for optimal memory access patterns.
    
    This function is adapted from FlashInfer's test code:
    Source: https://github.com/flashinfer-ai/flashinfer/blob/main/tests/gemm/test_groupwise_scaled_gemm_mxfp4.py
    See the `swizzle_blockscale()` function in that file.
    
    Args:
        unswizzled_sf: Scale factors with shape (b, m, n // sf_vec_size)
        b: Batch dimension
        m: M dimension (rows)
        n: K dimension (for scale computation, k_padded)
        sf_vec_size: Scale factor vector size (default 32)
    
    Returns:
        Swizzled scale factors with same shape but different memory layout
    """
    padded_input_sf_chunked = [
        _pad_scale_factors(unswizzled_sf[i], m, n, sf_vec_size) for i in range(b)
    ]
    padded_input_sf = torch.stack(padded_input_sf_chunked)
    major, minor = get_compute_capability(unswizzled_sf.device)
    out = get_fp4_quantization_module(f"{major}{minor}").block_scale_interleave_sm100(
        padded_input_sf
    )
    return out.view(padded_input_sf.shape)


def _quantize_e2m1(x: torch.Tensor) -> torch.Tensor:
    """
    Quantize tensor to FP4 E2M1 format and pack into uint8.
    
    FP4 E2M1 format has 1 sign bit, 2 exponent bits, and 1 mantissa bit.
    Representable values: 0, 0.5, 1, 1.5, 2, 3, 4, 6 (and negatives)
    
    Args:
        x: Input tensor to quantize
    
    Returns:
        Packed uint8 tensor with two FP4 values per byte
    """
    x = x.clamp(-6, 6)
    x_sign_bit = torch.lt(x, 0)
    x_abs = torch.abs(x)
    log_x_quant = torch.floor(torch.log2(x_abs)).clamp(0, 2)
    x_quant_e_fp32 = torch.exp2(log_x_quant)
    m_scale = 2
    x_quant_m_scaled_fp32 = torch.round(x_abs * m_scale / x_quant_e_fp32)
    mask = torch.ge(x_quant_m_scaled_fp32, m_scale)
    x_quant_data_raw_e = log_x_quant + mask
    x_quant_data_raw_m = x_quant_m_scaled_fp32 - mask * m_scale
    x_quant_data_raw = (
        x_sign_bit * 8 + x_quant_data_raw_e * m_scale + x_quant_data_raw_m
    ).to(torch.uint8)
    # Pack two FP4 values into one uint8
    x_quant_data = x_quant_data_raw[..., ::2] + x_quant_data_raw[..., 1::2] * 16
    return x_quant_data


def _dequantize_e2m1(x: torch.Tensor) -> torch.Tensor:
    """
    Dequantize tensor from FP4 E2M1 packed format.
    
    Args:
        x: Packed uint8 tensor with two FP4 values per byte
    
    Returns:
        Dequantized float32 tensor
    """
    x_quant_data_raw_1 = x % 16
    x_quant_data_raw_2 = x // 16
    x_quant_data_raw = torch.stack([x_quant_data_raw_1, x_quant_data_raw_2], dim=-1).flatten(start_dim=-2)
    x_sign_bit = x_quant_data_raw // 8
    x = x_quant_data_raw % 8
    m_scale = 2
    x_quant_data_raw_e = x // m_scale
    x_quant_data_raw_m = x % m_scale
    mask = torch.gt(x_quant_data_raw_e, 0).to(torch.float32)
    log_x_quant = x_quant_data_raw_e - mask
    x_quant_m_scaled_fp32 = x_quant_data_raw_m + mask * m_scale
    x_dequant_abs = x_quant_m_scaled_fp32 / m_scale * torch.exp2(log_x_quant)
    x_dequant = (0.5 - x_sign_bit) * 2 * x_dequant_abs
    return x_dequant


def _quantize_tensor_mxfp(
    x: torch.Tensor, 
    tile_size: int, 
    n_padded: int | None, 
    k_padded: int, 
    is_fp4: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize tensor to MXFP4 or MXFP8 format with per-tile scaling.
    
    This follows FlashInfer's quantization pattern exactly.
    
    Args:
        x: Input tensor to quantize
        tile_size: Size of each quantization tile (typically 32)
        n_padded: Padded N dimension (or None to skip N padding)
        k_padded: Padded K dimension
        is_fp4: If True, quantize to FP4 E2M1; if False, quantize to FP8 E4M3
    
    Returns:
        Tuple of (quantized_data, scale_data)
    """
    ue8m0_bias = 127
    if is_fp4:
        quant_amax = torch.tensor(6, dtype=torch.float32, device=x.device)
    else:
        fp8_info = torch.finfo(torch.float8_e4m3fn)
        quant_amax = torch.tensor(fp8_info.max, dtype=torch.float32, device=x.device)

    # Pad n dimension if needed
    if n_padded is not None and x.shape[-2] != n_padded:
        x = torch.cat([
            x, 
            torch.zeros((*x.shape[:-2], n_padded - x.shape[-2], x.shape[-1]), dtype=x.dtype, device=x.device)
        ], dim=-2)
    # Pad k dimension if needed
    if x.shape[-1] != k_padded:
        x = torch.cat([
            x, 
            torch.zeros((*x.shape[:-1], k_padded - x.shape[-1]), dtype=x.dtype, device=x.device)
        ], dim=-1)

    # Tile and compute per-tile scales
    x_tiled = x.unflatten(-1, (-1, tile_size))
    x_tiled_abs = x_tiled.abs()
    log2_x_scale = (
        torch.floor(torch.log2(x_tiled_abs.amax(dim=-1))) - torch.floor(torch.log2(quant_amax))
    ).clamp(-ue8m0_bias, ue8m0_bias)

    # Scale and quantize
    x_tiled_quant = (
        torch.exp2(torch.log2(x_tiled_abs) - log2_x_scale[..., None]).clamp(0, quant_amax) * x_tiled.sign()
    )
    x_quant = x_tiled_quant.flatten(-2, -1)

    if is_fp4:
        x_quant_data = _quantize_e2m1(x_quant)
    else:
        x_quant_data = x_quant.to(torch.float8_e4m3fn)
    x_scale_data = (log2_x_scale + ue8m0_bias).to(torch.uint8)

    return x_quant_data, x_scale_data


def _gemm_mxfp4_reference(
    A: torch.Tensor, 
    B: torch.Tensor, 
    As: torch.Tensor, 
    Bs: torch.Tensor, 
    tile_size: int, 
    n: int, 
    k: int, 
    output_dtype: torch.dtype = torch.bfloat16
) -> torch.Tensor:
    """
    Reference GEMM implementation for MXFP8 x MXFP4 with groupwise scaling.
    
    Computes: C = (A * As) @ (B * Bs).T
    
    Args:
        A: FP8 activations with shape (m, k_padded)
        B: Packed FP4 weights with shape (n_padded, k_padded // 2)
        As: A scale factors with shape (m, k_padded // tile_size)
        Bs: B scale factors with shape (n_padded, k_padded // tile_size)
        tile_size: Quantization tile size
        n: Actual N dimension (before padding)
        k: Actual K dimension (before padding)
        output_dtype: Output data type
    
    Returns:
        GEMM result with shape (m, n)
    """
    ue8m0_bias = 127
    A_f32 = A.to(torch.float32)
    B_f32 = _dequantize_e2m1(B)
    
    A_f32_reshape = rearrange(A_f32, "m (k b) -> m k b", b=tile_size)
    A_f32_scale_reshape = A_f32_reshape * rearrange(
        torch.exp2(As.to(torch.float32) - ue8m0_bias), "m k -> m k 1"
    )
    A_f32_scale = rearrange(A_f32_scale_reshape, "m k b -> m (k b)")[:, :k]
    
    B_f32_reshape = rearrange(B_f32, "n (k b) -> n k b", b=tile_size)
    B_f32_scale_reshape = B_f32_reshape * rearrange(
        torch.exp2(Bs.to(torch.float32) - ue8m0_bias), "n k -> n k 1"
    )
    B_f32_scale = rearrange(B_f32_scale_reshape, "n k b -> n (k b)")[:n, :k]
    
    return einsum(A_f32_scale, B_f32_scale, "m k, n k -> m n").to(output_dtype)


def _sanity_check_mxfp4_gemm(device: str = "cuda"):
    """
    Sanity check to verify FlashInfer MXFP4 GEMM produces correct results.
    
    This function follows FlashInfer's official test pattern exactly:
    https://github.com/flashinfer-ai/flashinfer/blob/main/tests/gemm/test_groupwise_scaled_gemm_mxfp4.py
    
    Key insights:
    1. Both a_scale and b_scale must be "swizzled" using block_scale_interleave_sm100
    2. a_scale requires additional padding per group to multiples of 128
    3. The quantization uses per-tile (32 elements) scaling with UE8M0 scale format
    
    Note: group_gemm_mxfp4_nt_groupwise is only supported on NVIDIA Blackwell architecture.
    """
    print("=" * 60)
    print("MXFP4 GEMM Sanity Check (FlashInfer pattern)")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("CUDA not available, skipping MXFP4 GEMM sanity check")
        return

    # Test parameters
    m = 128  # M dimension per group (must be multiple of 4)
    n = 256  # N dimension (output features)
    k = 256  # K dimension (input features)
    group_size = 2  # Number of groups/batches
    tile_size = 32  # Quantization tile size
    
    # Alignment requirements
    alignment_n = 8
    alignment_k = 128
    alignment_m_sf = 128  # Critical: a_scale padding alignment

    n_padded = ((n + alignment_n - 1) // alignment_n) * alignment_n
    k_padded = ((k + alignment_k - 1) // alignment_k) * alignment_k

    print(f"\nDimensions: m={m}, n={n}, k={k}, group_size={group_size}")
    print(f"Padded: n_padded={n_padded}, k_padded={k_padded}")

    # Generate random test data
    torch.manual_seed(42)
    a_val = torch.randn((group_size * m, k), dtype=torch.float32, device=device)
    b_val = torch.randn((group_size, n, k), dtype=torch.float32, device=device) / math.sqrt(k)

    print(f"\nInput shapes: a_val={a_val.shape}, b_val={b_val.shape}")

    # Quantize A (activations) to MXFP8
    print("\n--- Quantizing ---")
    a_fp8, a_scale = _quantize_tensor_mxfp(a_val, tile_size, None, k_padded, is_fp4=False)
    b_fp4, b_scale = _quantize_tensor_mxfp(b_val, tile_size, n_padded, k_padded, is_fp4=True)
    print(f"a_fp8: {a_fp8.shape}, a_scale: {a_scale.shape}")
    print(f"b_fp4: {b_fp4.shape}, b_scale: {b_scale.shape}")

    # Swizzle scales (CRITICAL for correct results!)
    print("\n--- Swizzling scales ---")
    a_scale_swizzled = _swizzle_blockscale(
        a_scale.unflatten(0, (group_size, m)), group_size, m, k_padded, tile_size
    ).flatten(0, 1)
    b_scale_swizzled = _swizzle_blockscale(b_scale, group_size, n_padded, k_padded, tile_size)
    print(f"a_scale_swizzled: {a_scale_swizzled.shape}")
    print(f"b_scale_swizzled: {b_scale_swizzled.shape}")

    # Create m_indptr for group GEMM
    group_arange = torch.arange(0, group_size + 1, dtype=torch.int32, device=device)
    m_indptr = group_arange * m
    print(f"\nm_indptr: {m_indptr.tolist()}")

    # Pad a_scale for group GEMM (CRITICAL!)
    # Each group's scale factors must be padded to multiples of alignment_m_sf
    print("\n--- Padding a_scale for group GEMM ---")
    m_indptr_padded = ((m_indptr + group_arange * (alignment_m_sf - 1)) // alignment_m_sf * alignment_m_sf)
    m_sf = m_indptr_padded[1:] - m_indptr_padded[:-1]
    print(f"m_indptr_padded: {m_indptr_padded.tolist()}")
    print(f"m_sf (scale sizes per group): {m_sf.tolist()}")

    a_scale_chunked = a_scale_swizzled.chunk(group_size, dim=0)
    a_scale_chunked = [
        torch.cat([x, torch.zeros(m_sf[i].item() - x.shape[0], *x.shape[1:], dtype=x.dtype, device=x.device)])
        for i, x in enumerate(a_scale_chunked)
    ]
    a_scale_final = torch.cat(a_scale_chunked)
    print(f"a_scale_final: {a_scale_final.shape}")

    # Compute reference result
    print("\n--- Computing reference ---")
    out_ref = torch.empty((group_size * m, n), dtype=torch.bfloat16, device=device)
    for i in range(group_size):
        out_ref[m * i : m * (i + 1)] = _gemm_mxfp4_reference(
            a_fp8[m * i : m * (i + 1)],
            b_fp4[i],
            a_scale[m * i : m * (i + 1)],
            b_scale[i],
            tile_size, n, k, torch.bfloat16
        )
    print(f"Reference shape: {out_ref.shape}")
    print(f"Reference sample: {out_ref[0, :5].tolist()}")

    # Run FlashInfer GEMM
    print("\n--- Running FlashInfer group_gemm_mxfp4_nt_groupwise ---")
    try:
        out = group_gemm_mxfp4_nt_groupwise(
            a_fp8, b_fp4,
            a_scale_final, b_scale_swizzled,
            m_indptr,
            mma_sm=1, tile_m=128, tile_n=128, tile_k=128,
            swap_ab=True, out_dtype=torch.bfloat16
        )[:, :n]

        print(f"FlashInfer output shape: {out.shape}")
        print(f"FlashInfer sample: {out[0, :5].tolist()}")

        # Compare results
        diff = (out.float() - out_ref.float()).abs()
        print(f"\n--- Comparison ---")
        print(f"Max abs diff: {diff.max().item():.6f}")
        print(f"Mean abs diff: {diff.mean().item():.6f}")

        # Check tolerance (same as FlashInfer test: atol=1e-2, rtol=1e-2)
        try:
            torch.testing.assert_close(out, out_ref, atol=1e-2, rtol=1e-2)
            print("\n✓ PASS: Results match within tolerance!")
            print("=" * 60)
            exit(0)
        except AssertionError as e:
            print(f"\n✗ FAIL: {e}")
            print("=" * 60)
            exit(1)

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        print("\nNote: group_gemm_mxfp4_nt_groupwise requires NVIDIA Blackwell GPU.")
        print("=" * 60)
        exit(1)


class GptOssModel(nn.Module):
    """GPT OSS model with FlashInfer support."""

    def __init__(self, config: GptOssArch):
        """Initialize the GPT OSS model."""
        super().__init__()

        # NOTE: Sanity check disabled - native MXFP4 now integrated into GptOssExperts
        # _sanity_check_mxfp4_gemm(device=config.device)
        self.config = config

        self.embed_tokens = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            padding_idx=0,
            device=torch.device(config.device),
            dtype=config.dtype,
        )
        self.layers = nn.ModuleList(
            [
                GptOssDecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_layers)
            ]
        )
        self.norm = GptOssRMSNorm(
            config.hidden_size,
            device=config.device,
        )
        self.sliding_window = config.sliding_window

    def forward(
        self,
        input_embeds: torch.Tensor,
        position_ids: torch.Tensor,
        qo_indptr: torch.Tensor,
        kv_cache_at_layer: torch.Tensor,
        kv_page_indices: torch.Tensor,
        kv_page_indptr: torch.Tensor,
        kv_last_page_lens: torch.Tensor,
        custom_mask: torch.Tensor,
        single_token_inference_mode: bool,
        adapter_subpass: AdapterSubpass | None,
    ) -> torch.Tensor:
        """Forward pass through the GPT OSS model."""

        # The current naive implementation does not distinguish between
        # single-token inference mode and batch inference mode
        _ = single_token_inference_mode

        hidden_states = input_embeds
        n, _ = hidden_states.size()

        page_size = kv_cache_at_layer[0].shape[2]

        batch_indices, batch_positions = ops.get_batch_indices_positions(
            append_indptr=qo_indptr,
            seq_lens=ops.get_seq_lens(kv_page_indptr, kv_last_page_lens, page_size),
            nnz=n,
        )

        batch_num = len(qo_indptr) - 1

        full_mask = custom_mask
        window_mask = custom_mask.clone()

        # For window attention layers, set the mask to 0 for positions that are
        # outside the sliding window.
        mask_offset = 0
        for batch_idx in range(batch_num):
            q_start = qo_indptr[batch_idx]
            q_end = qo_indptr[batch_idx + 1]

            kv_page_start = kv_page_indptr[batch_idx]
            kv_page_end = kv_page_indptr[batch_idx + 1]

            query_len = int(q_end - q_start)
            seq_len = int(
                (kv_page_end - kv_page_start - 1) * page_size
                + kv_last_page_lens[batch_idx]
            )
            mask_len = seq_len * query_len

            mask = window_mask[mask_offset : mask_offset + mask_len]
            mask_offset += mask_len

            mask = mask.view(query_len, seq_len)

            pos_id = position_ids[q_start:q_end]
            for q_idx in range(query_len):
                mask[
                    q_idx, : max(0, int(pos_id[q_idx]) - (self.sliding_window - 1))
                ] = 0

        full_mask = torch.where(
            full_mask,
            torch.tensor(0.0, dtype=input_embeds.dtype, device=input_embeds.device),
            torch.tensor(
                float("-inf"), dtype=input_embeds.dtype, device=input_embeds.device
            ),
        )
        window_mask = torch.where(
            window_mask,
            torch.tensor(0.0, dtype=input_embeds.dtype, device=input_embeds.device),
            torch.tensor(
                float("-inf"), dtype=input_embeds.dtype, device=input_embeds.device
            ),
        )

        for decoder_layer in self.layers:

            layer_outputs = decoder_layer(
                hidden_states=hidden_states,
                position_ids=position_ids,
                qo_indptr=qo_indptr,
                kv_cache_at_layer=kv_cache_at_layer,
                kv_page_indices=kv_page_indices,
                kv_page_indptr=kv_page_indptr,
                kv_last_page_lens=kv_last_page_lens,
                batch_indices=batch_indices,
                batch_positions=batch_positions,
                full_mask=full_mask,
                window_mask=window_mask,
                adapter_subpass=adapter_subpass,
            )

            hidden_states = layer_outputs

        hidden_states = self.norm(hidden_states)

        return hidden_states


class GptOssForCausalLM(nn.Module):
    """GPT OSS model for causal language modeling."""

    def __init__(self, config: GptOssArch):
        """Initialize the GPT OSS causal LM model."""
        super().__init__()
        self.config = config
        self.model = GptOssModel(config)
        self.lm_head = nn.Linear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
            device=torch.device(config.device),
            dtype=config.dtype,
        )

    def forward(self):
        """
        Should not be called. Method 'forward' is abstract in class
        'torch.nn.modules.module' so must be overridden in child class.
        """
        raise NotImplementedError("Should not be called")
