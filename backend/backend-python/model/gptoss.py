"""GPT OSS Large Language Model Architecture"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import flashinfer as ops
from flashinfer.gemm import group_gemm_mxfp4_nt_groupwise
from flashinfer.fp4_quantization import mxfp4_quantize
from adapter_utils import AdapterSubpass
from model.config import CommonArch, ModelConfig
from model.gptoss_utils import (
    FP4_VALUES,
    chunked_enumerate,
    GptOssRMSNorm,
    GptOssRotaryEmbedding,
)
from einops import einsum, rearrange


def mxfp8_quantize_blockwise(
    x: torch.Tensor, block_size: int = 32
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a tensor to FP8 with blockwise E8M0 scales.

    This follows the MXFP8 format where:
    - true_value = fp8_stored * 2^(scale - 127)
    
    The scale represents the shared exponent for the block. We store the FP8 mantissa
    values and separately track the block exponent.

    Args:
        x: Input tensor of shape [..., K] where K is divisible by block_size
        block_size: Size of blocks for computing scales (default 32)

    Returns:
        Tuple of (fp8_data, scales):
        - fp8_data: Quantized tensor in float8_e4m3fn format, same shape as x
        - scales: E8M0 scale factors as uint8, shape [..., K // block_size]
    """
    orig_shape = x.shape
    K = orig_shape[-1]
    assert K % block_size == 0, f"Last dimension {K} must be divisible by {block_size}"

    # Reshape to [..., num_blocks, block_size]
    num_blocks = K // block_size
    x_blocks = x.view(*orig_shape[:-1], num_blocks, block_size)

    # Compute max absolute value per block
    max_abs = x_blocks.abs().amax(dim=-1)  # [..., num_blocks]

    # For MXFP format, the shared exponent is ceil(log2(max_abs))
    # This ensures all values in the block fit when divided by 2^exponent
    # E8M0 stores exponent with bias 127: stored = exponent + 127
    eps = 1e-12
    log2_max = torch.log2(max_abs.clamp(min=eps))
    
    # Compute shared exponent (unbiased)
    exponent_unbiased = torch.ceil(log2_max).to(torch.int32).clamp(min=-127, max=128)
    
    # Convert to biased E8M0 format
    exponent_biased = (exponent_unbiased + 127).clamp(min=0, max=255)
    scales = exponent_biased.to(torch.uint8)

    # Scale input: fp8_value = x / 2^exponent = x * 2^(-exponent)
    # So that true_value = fp8_value * 2^exponent = x
    scale_power = (-exponent_unbiased).unsqueeze(-1)  # [..., num_blocks, 1]
    x_scaled = torch.ldexp(x_blocks.float(), scale_power.expand_as(x_blocks))

    # Reshape back to original shape and cast to FP8
    x_scaled = x_scaled.view(orig_shape)
    x_fp8 = x_scaled.to(torch.float8_e4m3fn)

    return x_fp8, scales


def verify_flashinfer_mxfp4_api():
    """Verify FlashInfer's MXFP4 GEMM API works correctly with our setup.
    
    Key findings from debugging:
    1. Must quantize each expert's weights INDEPENDENTLY (not batch-quantize and slice)
    2. Must quantize each expert's inputs INDEPENDENTLY  
    3. Use single-expert GEMM calls in a loop (grouped GEMM has segment issues)
    
    Raises:
        RuntimeError: If the outputs don't match within tolerance.
    """
    print("[MXFP4 API Test] Verifying FlashInfer MXFP4 API...")
    
    device = torch.device("cuda")
    dtype = torch.bfloat16
    
    # Test dimensions
    num_experts = 2
    M = 8  # tokens per expert (multiple of 4)
    K = 128  # input dimension (multiple of 128)
    N = 64  # output dimension
    total_tokens = num_experts * M
    
    # Create test data
    x = torch.randn(total_tokens, K, dtype=dtype, device=device) * 2.0
    w = torch.randn(num_experts, N, K, dtype=dtype, device=device) * 0.5
    
    # Reference computation
    reference_output = torch.zeros(total_tokens, N, dtype=dtype, device=device)
    for i in range(num_experts):
        start, end = i * M, (i + 1) * M
        reference_output[start:end] = x[start:end] @ w[i].T
    
    output_scale = reference_output.abs().mean().item()
    
    # Quantize each expert's weights INDEPENDENTLY (critical!)
    w_fp4_data_list = []
    w_fp4_scales_list = []
    for i in range(num_experts):
        expert_w_data, expert_w_scales = mxfp4_quantize(w[i])
        w_fp4_data_list.append(expert_w_data.unsqueeze(0))
        w_fp4_scales_list.append(expert_w_scales.unsqueeze(0))
    
    # Run single-expert GEMM loop (critical: don't use grouped GEMM)
    loop_output = torch.zeros(total_tokens, N, dtype=dtype, device=device)
    for i in range(num_experts):
        start, end = i * M, (i + 1) * M
        
        # Quantize input per-expert (critical!)
        expert_x_fp8, expert_x_scales = mxfp8_quantize_blockwise(x[start:end], block_size=32)
        expert_m_indptr = torch.tensor([0, M], dtype=torch.int32, device=device)
        
        expert_output = group_gemm_mxfp4_nt_groupwise(
            expert_x_fp8,
            w_fp4_data_list[i],
            expert_x_scales,
            w_fp4_scales_list[i],
            expert_m_indptr,
            out_dtype=dtype,
        )
        
        if torch.isnan(expert_output).any() or torch.isinf(expert_output).any():
            raise RuntimeError(f"[MXFP4 API Test] FAILED: Expert {i} output has NaN/Inf!")
        
        loop_output[start:end] = expert_output
    
    # Check error tolerance
    loop_abs_diff = (loop_output.float() - reference_output.float()).abs()
    loop_normalized_error = loop_abs_diff.mean().item() / max(output_scale, 1e-6)
    
    # Tolerance: FP4+FP8 vs raw float has inherent quantization error
    tolerance = 1.0
    if loop_normalized_error > tolerance:
        raise RuntimeError(
            f"[MXFP4 API Test] FAILED: Loop output mismatch exceeds tolerance!\n"
            f"  Loop normalized error: {loop_normalized_error:.4f} > tolerance: {tolerance}"
        )
    
    print(f"[MXFP4 API Test] PASSED! Normalized error: {loop_normalized_error:.2%}")


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
        # Weights are now stored in their native MXFP4 quantized format (blocks + scales)
        # and used directly with FlashInfer's native MXFP4 GEMM operations.
        # No fusion or dequantization is needed - weights are loaded directly as-is.

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
    """GPT OSS Experts layer containing the actual expert parameters.

    Weights are stored in MXFP4 quantized format (blocks + scales) and used
    directly with FlashInfer's group_gemm_mxfp4_nt_groupwise for native MXFP4 GEMM.
    """

    # MXFP4 uses a block size of 32 elements per scale factor
    # Each block of 32 elements is packed into 16 bytes (2 elements per byte)
    MXFP4_BLOCK_SIZE = 32
    MXFP4_BYTES_PER_BLOCK = 16  # 32 elements / 2 elements per byte

    # Type hints for registered buffers (stored MXFP4 format)
    gate_up_proj_blocks: torch.Tensor
    gate_up_proj_scales: torch.Tensor
    down_proj_blocks: torch.Tensor
    down_proj_scales: torch.Tensor

    # Class variable to track if API verification has been done
    _api_verified = False

    def __init__(self, config: GptOssArch):
        """Initialize the GPT OSS Experts layer."""
        super().__init__()
        
        # Run API verification once before first model load
        if not GptOssExperts._api_verified:
            verify_flashinfer_mxfp4_api()
            GptOssExperts._api_verified = True
        self.config = config
        self.num_experts = config.num_experts
        self.swiglu_limit = config.swiglu_limit
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        # Gate-up projection dequantized shape: (num_experts, intermediate_size * 2, hidden_size)
        # MXFP4 format stores blocks with shape: (num_experts, intermediate_size * 2, num_groups, bytes_per_group)
        # where num_groups = hidden_size // 32, bytes_per_group = 16
        # For group_gemm_mxfp4_nt_groupwise, weights need shape: (num_experts, n, k // 2)
        # Stored blocks can be reshaped: (num_experts, n, k//32, 16) -> (num_experts, n, k//2)
        num_gate_up_groups = config.hidden_size // self.MXFP4_BLOCK_SIZE
        gate_up_blocks_shape = (
            config.num_experts,
            config.intermediate_size * 2,
            num_gate_up_groups,
            self.MXFP4_BYTES_PER_BLOCK,
        )
        self.register_buffer(
            "gate_up_proj_blocks",
            torch.empty(
                gate_up_blocks_shape,
                device=torch.device(config.device),
                dtype=torch.uint8,
            ),
        )
        # MXFP4 scales: one scale per block of 32 elements
        # Shape: (num_experts, intermediate_size * 2, num_groups)
        gate_up_scales_shape = (
            config.num_experts,
            config.intermediate_size * 2,
            num_gate_up_groups,
        )
        self.register_buffer(
            "gate_up_proj_scales",
            torch.empty(
                gate_up_scales_shape,
                device=torch.device(config.device),
                dtype=torch.uint8,
            ),
        )
        self.gate_up_proj_bias = nn.Parameter(
            torch.empty(
                (config.num_experts, config.intermediate_size * 2),
                device=torch.device(config.device),
                dtype=config.dtype,
            )
        )

        # Down projection dequantized shape: (num_experts, hidden_size, intermediate_size)
        # MXFP4 format stores blocks with shape: (num_experts, hidden_size, num_groups, bytes_per_group)
        # where num_groups = intermediate_size // 32, bytes_per_group = 16
        num_down_groups = config.intermediate_size // self.MXFP4_BLOCK_SIZE
        down_blocks_shape = (
            config.num_experts,
            config.hidden_size,
            num_down_groups,
            self.MXFP4_BYTES_PER_BLOCK,
        )
        self.register_buffer(
            "down_proj_blocks",
            torch.empty(
                down_blocks_shape,
                device=torch.device(config.device),
                dtype=torch.uint8,
            ),
        )
        # MXFP4 scales: one scale per block of 32 elements
        # Shape: (num_experts, hidden_size, num_groups)
        down_scales_shape = (
            config.num_experts,
            config.hidden_size,
            num_down_groups,
        )
        self.register_buffer(
            "down_proj_scales",
            torch.empty(
                down_scales_shape,
                device=torch.device(config.device),
                dtype=torch.uint8,
            ),
        )
        self.down_proj_bias = nn.Parameter(
            torch.empty(
                (config.num_experts, config.hidden_size),
                device=torch.device(config.device),
                dtype=config.dtype,
            )
        )

    def _dequantize_mxfp4_weights(
        self, blocks: torch.Tensor, scales: torch.Tensor, dtype: torch.dtype
    ) -> torch.Tensor:
        """Dequantize MXFP4 weights to floating point.

        Args:
            blocks: Packed FP4 values, shape [..., num_groups, bytes_per_group]
            scales: Scale factors (uint8 E8M0 biased by 127), shape [..., num_groups]
            dtype: Target dtype for output

        Returns:
            Dequantized weights, shape [..., num_groups * bytes_per_group * 2]
        """
        # Convert scales from biased E8M0 to exponent
        scales_exp = scales.to(torch.int32) - 127

        *prefix_shape, g, b = blocks.shape
        rows_total = 1
        for dim in prefix_shape:
            rows_total *= dim
        rows_total *= g

        blocks_flat = blocks.reshape(rows_total, b)
        scales_flat = scales_exp.reshape(rows_total, 1)

        # Extract low and high 4-bit indices
        idx_lo = (blocks_flat & 0x0F).to(torch.long)
        idx_hi = (blocks_flat >> 4).to(torch.long)

        # FP4 lookup table
        lut = torch.tensor(FP4_VALUES, dtype=dtype, device=blocks.device)

        # Create output tensor and populate
        out = torch.empty(rows_total, b * 2, dtype=dtype, device=blocks.device)
        out[:, 0::2] = lut[idx_lo]  # Low 4-bit values at even indices
        out[:, 1::2] = lut[idx_hi]  # High 4-bit values at odd indices

        # Apply scale factors (2^scale)
        torch.ldexp(out, scales_flat, out=out)

        return out.reshape(*prefix_shape, g, b * 2).view(*prefix_shape, g * b * 2)

    def _group_gemm_mxfp4(
        self,
        x: torch.Tensor,
        expert_indices: torch.Tensor,
        w_blocks: torch.Tensor,
        w_scales: torch.Tensor,
        bias: torch.Tensor,
        in_dim: int,
        out_dim: int,
    ) -> torch.Tensor:
        """Perform grouped MXFP4 GEMM using FlashInfer's group_gemm_mxfp4_nt_groupwise.

        This function groups tokens by their assigned expert and performs
        a single fused GEMM operation for all experts.

        Args:
            x: Input tensor of shape [batch, experts_per_token, in_dim]
            expert_indices: Expert indices of shape [batch, experts_per_token]
            w_blocks: Weight blocks of shape [num_experts, out_dim, k//32, 16]
            w_scales: Weight scales of shape [num_experts, out_dim, k//32]
            bias: Bias tensor of shape [num_experts, out_dim]
            in_dim: Input dimension (K)
            out_dim: Output dimension (N)

        Returns:
            Output tensor of shape [batch, experts_per_token, out_dim]
        """
        # Alignment requirements for group_gemm_mxfp4_nt_groupwise
        K_ALIGN = 128  # K dimension must be multiple of tile_k (default 128)
        M_ALIGN = 4    # M segments must be multiples of 4

        batch_size, experts_per_token, _ = x.shape
        input_dtype = x.dtype
        device = x.device
        total_tokens = batch_size * experts_per_token

        # Pad K dimension if needed
        k_padded = ((in_dim + K_ALIGN - 1) // K_ALIGN) * K_ALIGN
        k_pad_amount = k_padded - in_dim

        # Flatten inputs
        x_flat = x.reshape(total_tokens, in_dim)  # [total_tokens, in_dim]
        indices_flat = expert_indices.reshape(total_tokens)  # [total_tokens]

        # Sort tokens by expert index for grouped GEMM
        sorted_indices = torch.argsort(indices_flat, stable=True)
        sorted_expert_ids = indices_flat[sorted_indices]
        x_sorted = x_flat[sorted_indices]  # [total_tokens, in_dim]

        # Compute m_indptr: boundaries for each expert's tokens
        # Each element must be a multiple of 4 for group_gemm_mxfp4_nt_groupwise
        expert_counts = torch.zeros(self.num_experts, dtype=torch.int32, device=device)
        for i in range(self.num_experts):
            expert_counts[i] = (sorted_expert_ids == i).sum()

        # Pad counts to multiples of M_ALIGN
        padded_counts = ((expert_counts + M_ALIGN - 1) // M_ALIGN) * M_ALIGN
        total_m_padded = int(padded_counts.sum().item())

        # Build m_indptr
        m_indptr = torch.zeros(self.num_experts + 1, dtype=torch.int32, device=device)
        m_indptr[1:] = torch.cumsum(padded_counts, dim=0)

        # Pad x_sorted to match padded counts AND padded K dimension
        x_padded = torch.zeros(total_m_padded, k_padded, dtype=input_dtype, device=device)
        orig_positions = []  # Track original positions for unpadding
        current_pos = 0
        sorted_pos = 0
        for i in range(self.num_experts):
            count = int(expert_counts[i].item())
            padded = int(padded_counts[i].item())
            if count > 0:
                # Copy original data (K dimension will be zero-padded implicitly)
                x_padded[current_pos : current_pos + count, :in_dim] = x_sorted[sorted_pos : sorted_pos + count]
                orig_positions.extend(range(current_pos, current_pos + count))
            sorted_pos += count
            current_pos += padded

        # NOTE: We quantize input per-expert below (not batch-quantize and slice)
        # Based on testing, batch quantization followed by slicing causes issues

        # Pad weight blocks and scales for K alignment if needed
        if k_pad_amount > 0:
            # Original: (num_experts, out_dim, k//32, 16) where k//32 * 16 * 2 = k (original)
            # Need to pad to k_padded
            # Number of new blocks needed: k_padded // 32 - in_dim // 32
            orig_k_blocks = in_dim // 32
            new_k_blocks = k_padded // 32
            pad_k_blocks = new_k_blocks - orig_k_blocks

            # Pad weight blocks: (num_experts, out_dim, k//32, 16) -> (num_experts, out_dim, k_padded//32, 16)
            w_blocks_padded = torch.zeros(
                self.num_experts, out_dim, new_k_blocks, 16,
                dtype=w_blocks.dtype, device=device
            )
            w_blocks_padded[:, :, :orig_k_blocks, :] = w_blocks

            # Pad weight scales: (num_experts, out_dim, k//32) -> (num_experts, out_dim, k_padded//32)
            # Zero scale (E8M0 with exponent 0) = 127 (bias)
            w_scales_padded = torch.full(
                (self.num_experts, out_dim, new_k_blocks),
                127,  # Zero scale in E8M0 format
                dtype=w_scales.dtype, device=device
            )
            w_scales_padded[:, :, :orig_k_blocks] = w_scales
        else:
            w_blocks_padded = w_blocks
            w_scales_padded = w_scales

        # Reshape weight blocks: (num_experts, out_dim, k_padded//32, 16) -> (num_experts, out_dim, k_padded//2)
        w_data = w_blocks_padded.view(self.num_experts, out_dim, k_padded // 2)

        # Perform MXFP4 GEMM for each expert separately
        # NOTE: Using loop of single-expert GEMMs because:
        # 1. Grouped GEMM has issues with multiple segments (second segment produces wrong output)
        # 2. Must quantize inputs and weights per-expert (batch quantize + slice causes inf)
        output_padded = torch.zeros(total_m_padded, out_dim, dtype=input_dtype, device=device)
        
        # Debug: check for issues on first call
        if not hasattr(self, '_debug_printed'):
            self._debug_printed = True
            print(f"[DEBUG _group_gemm_mxfp4] First call debug:")
            print(f"  x_padded: shape={x_padded.shape}, range=[{x_padded.min():.4f}, {x_padded.max():.4f}]")
            print(f"  w_data: shape={w_data.shape}")
            print(f"  w_scales_padded: shape={w_scales_padded.shape}")
            # Check weight stats per expert
            for i in range(min(2, self.num_experts)):
                print(f"  Expert {i} w_data: range=[{w_data[i].float().min():.4f}, {w_data[i].float().max():.4f}]")
                print(f"  Expert {i} w_scales: range=[{w_scales_padded[i].min()}, {w_scales_padded[i].max()}]")
        
        for expert_idx in range(self.num_experts):
            start = int(m_indptr[expert_idx].item())
            end = int(m_indptr[expert_idx + 1].item())
            
            if end > start:
                # Extract and quantize this expert's input SEPARATELY (not slice from batch)
                expert_x_padded = x_padded[start:end]
                expert_x_fp8, expert_x_scale = mxfp8_quantize_blockwise(expert_x_padded, block_size=32)
                
                # Get this expert's weights (these are loaded per-expert, so slicing is fine)
                expert_w_data = w_data[expert_idx:expert_idx+1]  # [1, out_dim, k_padded//2]
                expert_w_scales = w_scales_padded[expert_idx:expert_idx+1]  # [1, out_dim, k_padded//32]
                
                # Single-expert m_indptr
                expert_m = end - start
                expert_m_indptr = torch.tensor([0, expert_m], dtype=torch.int32, device=device)
                
                # Call FlashInfer for this expert
                expert_output = group_gemm_mxfp4_nt_groupwise(
                    expert_x_fp8,
                    expert_w_data,
                    expert_x_scale,
                    expert_w_scales,
                    expert_m_indptr,
                    out_dtype=input_dtype,
                )
                
                # Debug: check for NaN/Inf
                if torch.isnan(expert_output).any() or torch.isinf(expert_output).any():
                    nan_count = torch.isnan(expert_output).sum().item()
                    inf_count = torch.isinf(expert_output).sum().item()
                    print(f"[DEBUG] Expert {expert_idx} output has {nan_count} NaN, {inf_count} Inf!")
                    print(f"  Input x_padded: range=[{expert_x_padded.min():.4f}, {expert_x_padded.max():.4f}]")
                    print(f"  x_fp8: range=[{expert_x_fp8.float().min():.4f}, {expert_x_fp8.float().max():.4f}]")
                    print(f"  x_scale: range=[{expert_x_scale.min()}, {expert_x_scale.max()}]")
                    print(f"  w_data: shape={expert_w_data.shape}")
                    print(f"  w_scales: range=[{expert_w_scales.min()}, {expert_w_scales.max()}]")
                
                output_padded[start:end] = expert_output

        # Extract original (unpadded) outputs and unsort
        output_sorted = output_padded[orig_positions]  # [total_tokens, out_dim]

        # Add bias per expert
        for i in range(self.num_experts):
            mask = sorted_expert_ids == i
            if mask.any():
                output_sorted[mask] += bias[i]

        # Unsort to original order
        output_flat = torch.zeros(total_tokens, out_dim, dtype=input_dtype, device=device)
        output_flat[sorted_indices] = output_sorted

        # Reshape back to [batch, experts_per_token, out_dim]
        return output_flat.reshape(batch_size, experts_per_token, out_dim)

    def forward(self, t: torch.Tensor, expert_indices: torch.Tensor) -> torch.Tensor:
        """Forward pass through the experts using FlashInfer's group_gemm_mxfp4_nt_groupwise.

        Args:
            t: Input tensor of shape [batch, hidden_size]
            expert_indices: Expert indices of shape [batch, experts_per_token]

        Returns:
            Output tensor of shape [batch, experts_per_token, hidden_size]
        """
        batch_size, experts_per_token = expert_indices.shape

        # Expand input for each expert: [batch, experts_per_token, hidden_size]
        t_expanded = t.unsqueeze(1).expand(-1, experts_per_token, -1).contiguous()

        # Gate-up projection using grouped MXFP4 GEMM with FlashInfer
        # Input: [batch, experts_per_token, hidden_size]
        # Weights: [num_experts, intermediate_size * 2, hidden_size // 32, 16]
        # Output: [batch, experts_per_token, intermediate_size * 2]
        gate_up_output = self._group_gemm_mxfp4(
            t_expanded,
            expert_indices,
            self.gate_up_proj_blocks,
            self.gate_up_proj_scales,
            self.gate_up_proj_bias,
            in_dim=self.hidden_size,
            out_dim=self.intermediate_size * 2,
        )

        # Inline swiglu function
        x_glu, x_linear = gate_up_output[..., ::2], gate_up_output[..., 1::2]

        # Clamp the input values
        x_glu = x_glu.clamp(min=None, max=self.swiglu_limit)
        x_linear = x_linear.clamp(min=-self.swiglu_limit, max=self.swiglu_limit)
        out_glu = x_glu * torch.sigmoid(1.702 * x_glu)

        # Add an extra bias of 1 to the linear layer
        t = out_glu * (x_linear + 1)

        # Down projection using grouped MXFP4 GEMM with FlashInfer
        # Input: [batch, experts_per_token, intermediate_size]
        # Weights: [num_experts, hidden_size, intermediate_size // 32, 16]
        # Output: [batch, experts_per_token, hidden_size]
        down_output = self._group_gemm_mxfp4(
            t,
            expert_indices,
            self.down_proj_blocks,
            self.down_proj_scales,
            self.down_proj_bias,
            in_dim=self.intermediate_size,
            out_dim=self.hidden_size,
        )

        return down_output


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


class GptOssModel(nn.Module):
    """GPT OSS model with FlashInfer support."""

    def __init__(self, config: GptOssArch):
        """Initialize the GPT OSS model."""
        super().__init__()
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
