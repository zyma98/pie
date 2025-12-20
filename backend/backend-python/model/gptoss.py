"""GPT OSS Large Language Model Architecture"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
from torch import nn
import flashinfer as ops
from flashinfer import fp4_quantize
from flashinfer.autotuner import autotune
from flashinfer.fp4_quantization import block_scale_interleave
from flashinfer.fused_moe import trtllm_fp4_block_scale_moe
from flashinfer.fused_moe.core import (
    get_w2_permute_indices_with_cache,
    _maybe_get_cached_w3_w1_permute_indices,
)
from adapter_utils import AdapterSubpass
from model.config import CommonArch, ModelConfig
from model.gptoss_utils import (
    FP4_VALUES,
    chunked_enumerate,
    GptOssRMSNorm,
    GptOssRotaryEmbedding,
)
from einops import einsum, rearrange


# ====================================================================================
# FlashInfer Fused MoE Helper Functions
# ====================================================================================

# Alignment requirement for trtllm_fp4_block_scale_moe
ALIGNMENT = 256

# Max num tokens to tune for trtllm-gen fused moe
TUNE_MAX_NUM_TOKENS = 4096


def pad_to_multiple(size: int, multiple: int = ALIGNMENT) -> int:
    """Calculate padded size to be a multiple of the given number."""
    if size % multiple == 0:
        return size
    return ((size + multiple - 1) // multiple) * multiple


def deinterleave_gate_up_weights(weights: torch.Tensor) -> torch.Tensor:
    """
    De-interleave gate_up weights from GPT OSS format to FlashInfer format.
    
    GPT OSS stores gate_up as interleaved: [gate[0], linear[0], gate[1], linear[1], ...]
    FlashInfer expects non-interleaved: [linear[0], linear[1], ..., gate[0], gate[1], ...]
    
    Args:
        weights: Input tensor of shape [num_experts, 2*intermediate_size, hidden_size]
                 where even columns are gate, odd columns are linear (interleaved)
    
    Returns:
        De-interleaved tensor of shape [num_experts, 2*intermediate_size, hidden_size]
        where first half is linear, second half is gate
    """
    num_experts, fused_size, hidden_size = weights.shape
    intermediate_size = fused_size // 2
    
    # Extract interleaved parts: gate at even indices, linear at odd indices
    gate_part = weights[:, 0::2, :]   # [num_experts, intermediate_size, hidden_size]
    linear_part = weights[:, 1::2, :] # [num_experts, intermediate_size, hidden_size]
    
    # Concatenate: linear first, gate second (FlashInfer format)
    return torch.cat([linear_part, gate_part], dim=1)


def deinterleave_gate_up_bias(bias: torch.Tensor) -> torch.Tensor:
    """
    De-interleave gate_up bias from GPT OSS format to FlashInfer format.
    
    Args:
        bias: Input tensor of shape [num_experts, 2*intermediate_size]
              where even indices are gate, odd indices are linear (interleaved)
    
    Returns:
        De-interleaved tensor of shape [num_experts, 2*intermediate_size]
        where first half is linear, second half is gate
    """
    num_experts, fused_size = bias.shape
    
    # Extract interleaved parts
    gate_part = bias[:, 0::2]   # [num_experts, intermediate_size]
    linear_part = bias[:, 1::2] # [num_experts, intermediate_size]
    
    # Concatenate: linear first, gate second (FlashInfer format)
    return torch.cat([linear_part, gate_part], dim=1)


def pad_gate_up_weights(
    weights: torch.Tensor,
    padded_hidden_size: int,
    padded_intermediate_size: int,
) -> torch.Tensor:
    """
    Pad gate_up fused weights from [num_experts, 2*intermediate_size, hidden_size]
    to [num_experts, 2*padded_intermediate_size, padded_hidden_size].
    
    IMPORTANT: The first half is the linear part, the second half is the gate part.
    We need to pad each part separately, NOT pad all at the end of the fused matrix.
    """
    num_experts, fused_size, hidden_size = weights.shape
    intermediate_size = fused_size // 2
    
    if hidden_size == padded_hidden_size and intermediate_size == padded_intermediate_size:
        return weights
    
    # Split into linear and gate parts
    linear_part = weights[:, :intermediate_size, :]
    gate_part = weights[:, intermediate_size:, :]
    
    # Pad each part separately
    padded_linear = torch.zeros(
        (num_experts, padded_intermediate_size, padded_hidden_size),
        dtype=weights.dtype,
        device=weights.device,
    )
    padded_linear[:, :intermediate_size, :hidden_size] = linear_part
    
    padded_gate = torch.zeros(
        (num_experts, padded_intermediate_size, padded_hidden_size),
        dtype=weights.dtype,
        device=weights.device,
    )
    padded_gate[:, :intermediate_size, :hidden_size] = gate_part
    
    # Concatenate back: [linear_padded | gate_padded]
    return torch.cat([padded_linear, padded_gate], dim=1)


def pad_down_weights(
    weights: torch.Tensor,
    padded_hidden_size: int,
    padded_intermediate_size: int,
) -> torch.Tensor:
    """
    Pad down projection weights from [num_experts, hidden_size, intermediate_size]
    to [num_experts, padded_hidden_size, padded_intermediate_size].
    """
    num_experts, hidden_size, intermediate_size = weights.shape
    
    if hidden_size == padded_hidden_size and intermediate_size == padded_intermediate_size:
        return weights
    
    padded = torch.zeros(
        (num_experts, padded_hidden_size, padded_intermediate_size),
        dtype=weights.dtype,
        device=weights.device,
    )
    padded[:, :hidden_size, :intermediate_size] = weights
    return padded


def pad_gate_up_bias(
    bias: torch.Tensor,
    padded_intermediate_size: int,
) -> torch.Tensor:
    """
    Pad gate_up bias from [num_experts, 2*intermediate_size]
    to [num_experts, 2*padded_intermediate_size].
    
    IMPORTANT: Same as gate_up weights, pad each part separately.
    """
    num_experts, fused_size = bias.shape
    intermediate_size = fused_size // 2
    
    if intermediate_size == padded_intermediate_size:
        return bias
    
    # Split into linear and gate parts
    linear_part = bias[:, :intermediate_size]
    gate_part = bias[:, intermediate_size:]
    
    # Pad each part separately
    padded_linear = torch.zeros(
        (num_experts, padded_intermediate_size),
        dtype=bias.dtype,
        device=bias.device,
    )
    padded_linear[:, :intermediate_size] = linear_part
    
    padded_gate = torch.zeros(
        (num_experts, padded_intermediate_size),
        dtype=bias.dtype,
        device=bias.device,
    )
    padded_gate[:, :intermediate_size] = gate_part
    
    # Concatenate back: [linear_padded | gate_padded]
    return torch.cat([padded_linear, padded_gate], dim=1)


def pad_down_bias(
    bias: torch.Tensor,
    padded_hidden_size: int,
) -> torch.Tensor:
    """
    Pad down projection bias from [num_experts, hidden_size]
    to [num_experts, padded_hidden_size].
    """
    num_experts, hidden_size = bias.shape
    
    if hidden_size == padded_hidden_size:
        return bias
    
    padded = torch.zeros(
        (num_experts, padded_hidden_size),
        dtype=bias.dtype,
        device=bias.device,
    )
    padded[:, :hidden_size] = bias
    return padded


def quant_mxfp4_batches(
    a: torch.Tensor,
    is_sf_swizzled_layout: bool,
):
    """FP4 batch quantization function."""
    num_experts = a.shape[0]
    sf_vec_size = 32  # MXFP4 uses 32-element blocks

    quant_a = []
    sfs = []
    a_global_sf = torch.tensor(1.0, dtype=torch.float32).cuda()
    for i in range(num_experts):
        a_fp4, a_sf = fp4_quantize(
            a[i].cuda(), a_global_sf, sf_vec_size, True, is_sf_swizzled_layout
        )
        quant_a.append(a_fp4)
        sfs.append(a_sf)

    result_quant_a = torch.stack(quant_a)
    result_sfs = torch.stack(sfs)

    return result_quant_a, result_sfs


# ====================================================================================
# Reference MoE Implementation (for debugging/verification)
# ====================================================================================


def routing_reference(expert_logits: torch.Tensor, top_k: int, padding: int):
    """Reference routing implementation for permutation calculation."""
    original_device = expert_logits.device
    expert_logits = expert_logits.cpu()
    num_tokens, num_experts = expert_logits.shape
    assert top_k <= num_experts

    num_tokens_per_expert = torch.zeros(num_experts, dtype=torch.int64)
    expanded_token_idx_to_expert = -torch.ones(num_tokens * top_k, dtype=torch.int64)
    expanded_token_idx_to_idx_in_expert = -torch.ones(
        num_tokens * top_k, dtype=torch.int64
    )

    top_k_logits, top_k_indices = torch.topk(expert_logits, top_k, dim=1)
    for token_idx in range(num_tokens):
        for k in range(top_k):
            expanded_idx = token_idx * top_k + k
            expert_index = top_k_indices[token_idx, k]
            expanded_token_idx_to_expert[expanded_idx] = expert_index
            expanded_token_idx_to_idx_in_expert[expanded_idx] = num_tokens_per_expert[
                expert_index
            ]
            num_tokens_per_expert[expert_index] += 1

    def div_up_mul(a, b):
        return (a + b - 1) // b * b

    padded_tokens_per_expert_prefix_sum = torch.zeros(num_experts + 1, dtype=torch.int64)
    for ii in range(num_experts):
        padded_tokens_per_expert_prefix_sum[ii + 1] = (
            padded_tokens_per_expert_prefix_sum[ii]
            + div_up_mul(num_tokens_per_expert[ii], padding)
        )
    permuted_buffer_size = padded_tokens_per_expert_prefix_sum[num_experts]

    expanded_token_idx_to_permuted_idx = -torch.ones(
        num_tokens * top_k, dtype=torch.int64
    )
    permuted_idx_to_expanded_idx = -torch.ones(permuted_buffer_size, dtype=torch.int64)
    permuted_idx_to_token_idx = -torch.ones(permuted_buffer_size, dtype=torch.int64)

    for token_idx in range(num_tokens):
        for k in range(top_k):
            expanded_idx = token_idx * top_k + k
            expert = expanded_token_idx_to_expert[expanded_idx]
            offset_within_expert = expanded_token_idx_to_idx_in_expert[expanded_idx]
            offset_for_expert = padded_tokens_per_expert_prefix_sum[expert]
            permuted_idx = offset_for_expert + offset_within_expert

            expanded_token_idx_to_permuted_idx[expanded_idx] = permuted_idx
            permuted_idx_to_expanded_idx[permuted_idx] = expanded_idx
            permuted_idx_to_token_idx[permuted_idx] = token_idx

    return {
        "paddedTokensPerExpertPrefixSum": padded_tokens_per_expert_prefix_sum.to(
            original_device
        ),
        "permutedBufferSize": permuted_buffer_size.item(),
        "expandedTokenIdxToPermutedIdx": expanded_token_idx_to_permuted_idx.to(
            original_device
        ),
        "permutedIdxToExpandedIdx": permuted_idx_to_expanded_idx.to(original_device),
        "numTokensPerExpert": num_tokens_per_expert.to(original_device),
        "expandedTokenIdxToExpert": expanded_token_idx_to_expert.to(original_device),
        "topKLogits": top_k_logits.to(original_device),
        "permutedIdxToTokenIdx": permuted_idx_to_token_idx.to(original_device),
        "topKIndices": top_k_indices.to(original_device),
    }


def routing_reference_renormalize(
    expert_logits: torch.Tensor, top_k: int, num_experts: int, padding: int
):
    """TopK -> Softmax routing reference."""
    topk_values, topk_idx = torch.topk(expert_logits, k=top_k, dim=-1)
    topk_values = torch.nn.functional.softmax(topk_values.float(), dim=-1)

    new_mask = torch.zeros_like(expert_logits)
    new_mask.scatter_(-1, topk_idx, 1)
    scores = expert_logits * new_mask

    for i in range(topk_idx.shape[0]):
        for j in range(topk_idx.shape[1]):
            scores[i, topk_idx[i, j]] = topk_values[i, j]
    permute_info = routing_reference(scores, top_k, padding)
    return permute_info, scores


def run_moe_reference(
    num_tokens: int,
    num_experts: int,
    hidden_size: int,
    intermediate_size: int,
    top_k: int,
    padding: int,
    hidden_states: torch.Tensor,
    gemm1_weights: torch.Tensor,
    gemm2_weights: torch.Tensor,
    permute_info: dict,
    gemm1_bias: torch.Tensor | None = None,
    gemm2_bias: torch.Tensor | None = None,
    gemm1_alpha: torch.Tensor | None = None,
    gemm1_beta: torch.Tensor | None = None,
    gemm1_clamp_limit: torch.Tensor | None = None,
):
    """
    Reference MoE implementation (dequantized, for debugging).
    
    This matches the FlashInfer fused MoE kernel behavior.
    Note: gemm1_weights should be in non-interleaved format [linear..., gate...]
    """
    # Permute
    total_num_padded_tokens = permute_info["permutedBufferSize"]
    expanded_idx_to_permuted_idx = permute_info["expandedTokenIdxToPermutedIdx"].cpu()
    num_tokens_per_expert = permute_info["numTokensPerExpert"].cpu()

    permute_output = torch.full(
        (total_num_padded_tokens, hidden_size), float("nan"), device="cuda"
    ).to(torch.float)
    for i in range(num_tokens):
        for j in range(top_k):
            permuted_idx = expanded_idx_to_permuted_idx[i * top_k + j]
            permute_output[permuted_idx] = hidden_states[i]

    # Gemm1
    gemm1_output = torch.full(
        (total_num_padded_tokens, 2 * intermediate_size),
        float("nan"),
        device="cuda",
    ).to(torch.float)
    i = 0
    for expert_idx in range(num_experts):
        my_num_tokens = num_tokens_per_expert[expert_idx]
        if my_num_tokens == 0:
            continue
        my_a = permute_output[i : i + my_num_tokens]
        my_b = gemm1_weights[expert_idx]
        my_c = my_a @ my_b.t()
        if gemm1_bias is not None:
            my_c = my_c + gemm1_bias[expert_idx]
        gemm1_output[i : i + my_num_tokens] = my_c
        i += my_num_tokens
        i = (i + padding - 1) // padding * padding

    # Activation
    activation_output = torch.full(
        (total_num_padded_tokens, intermediate_size), float("nan"), device="cuda"
    ).to(torch.float)

    i = 0
    for expert_idx in range(num_experts):
        my_num_tokens = num_tokens_per_expert[expert_idx]
        if my_num_tokens == 0:
            continue
        my_a = gemm1_output[i : i + my_num_tokens]
        my_x1 = my_a[:, :intermediate_size]  # linear part
        my_x2 = my_a[:, intermediate_size:]  # gated part

        # Apply clamping if clamp_limit is provided (GPT OSS style)
        if gemm1_clamp_limit is not None:
            clamp_limit = gemm1_clamp_limit[expert_idx]
            my_x2 = my_x2.clamp(max=clamp_limit)
            my_x1 = my_x1.clamp(min=-clamp_limit, max=clamp_limit)

        # Apply activation with optional alpha scaling
        if gemm1_alpha is not None:
            alpha = gemm1_alpha[expert_idx]
            gated_output = my_x2 * torch.sigmoid(alpha * my_x2)
        else:
            gated_output = torch.nn.functional.silu(my_x2)

        # Apply beta offset to linear part
        if gemm1_beta is not None:
            beta = gemm1_beta[expert_idx]
            linear_output = my_x1 + beta
        else:
            linear_output = my_x1

        activation_output[i : i + my_num_tokens] = gated_output * linear_output
        i += my_num_tokens
        i = (i + padding - 1) // padding * padding

    # Convert activation to bf16 and back (matches kernel behavior)
    activation_output = activation_output.to(torch.bfloat16).to(torch.float)

    # Gemm2
    gemm2_output = torch.full(
        (total_num_padded_tokens, hidden_size), float("nan"), device="cuda"
    ).to(torch.float)
    i = 0
    for expert_idx in range(num_experts):
        my_num_tokens = num_tokens_per_expert[expert_idx]
        if my_num_tokens == 0:
            continue
        my_a = activation_output[i : i + my_num_tokens]
        my_b = gemm2_weights[expert_idx]
        my_c = my_a @ my_b.t()
        if gemm2_bias is not None:
            my_c = my_c + gemm2_bias[expert_idx]
        gemm2_output[i : i + my_num_tokens] = my_c
        i += my_num_tokens
        i = (i + padding - 1) // padding * padding

    # Finalize
    expert_weight = permute_info["topKLogits"].to(torch.float)
    finalize_output = torch.full(
        (num_tokens, hidden_size), float("nan"), device="cuda"
    ).to(torch.float)
    for i in range(num_tokens):
        acc = torch.zeros(hidden_size, dtype=torch.float, device="cuda")
        for top_k_idx in range(top_k):
            expanded_idx = i * top_k + top_k_idx
            permuted_idx = expanded_idx_to_permuted_idx[expanded_idx]
            original_vector = gemm2_output[permuted_idx]
            weight = expert_weight[i, top_k_idx]
            acc += original_vector * weight
        finalize_output[i] = acc

    return finalize_output


def prepare_moe_weights_for_kernel(
    gate_up_weights: torch.Tensor,
    down_weights: torch.Tensor,
    padded_hidden_size: int,
    padded_intermediate_size: int,
    num_experts: int,
    cache_permute_indices: Dict[tuple, torch.Tensor],
    gate_up_bias: torch.Tensor | None = None,
    down_bias: torch.Tensor | None = None,
) -> tuple:
    """
    Prepare MoE weights for FlashInfer fused MoE kernel.
    
    This includes:
    1. Quantizing weights to MXFP4
    2. Reshaping to proper formats
    3. Shuffling for transposed MMA output
    4. Shuffling bias vectors to match weight row reordering
    
    Args:
        gate_up_weights: Padded gate_up weights [num_experts, 2*padded_intermediate_size, padded_hidden_size]
        down_weights: Padded down weights [num_experts, padded_hidden_size, padded_intermediate_size]
        padded_hidden_size: Padded hidden size (multiple of 256)
        padded_intermediate_size: Padded intermediate size (multiple of 256)
        num_experts: Number of experts
        cache_permute_indices: Cache for permute indices
        gate_up_bias: Optional padded gate_up bias [num_experts, 2*padded_intermediate_size]
        down_bias: Optional padded down bias [num_experts, padded_hidden_size]
    
    Returns:
        Tuple of (gemm1_weights_shuffled, gemm1_scales_shuffled, 
                  gemm2_weights_shuffled, gemm2_scales_shuffled,
                  gemm1_bias_shuffled, gemm2_bias_shuffled)
    """
    epilogue_tile_m = 128

    # Quantize weights with swizzled layout (for dequant reference)
    gemm1_weights_quant, _ = quant_mxfp4_batches(gate_up_weights, True)
    gemm2_weights_quant, _ = quant_mxfp4_batches(down_weights, True)

    # Quantize weights with linear layout for kernels
    _, gemm1_scales_linear = quant_mxfp4_batches(gate_up_weights, False)
    _, gemm2_scales_linear = quant_mxfp4_batches(down_weights, False)

    # Convert quantized weights to proper formats
    gemm1_weights_fp4 = gemm1_weights_quant.view(torch.float8_e4m3fn).reshape(
        num_experts, 2 * padded_intermediate_size, padded_hidden_size // 2
    )
    gemm1_scales_linear_fp4 = gemm1_scales_linear.view(torch.float8_e4m3fn).reshape(
        num_experts, 2 * padded_intermediate_size, padded_hidden_size // 32
    )

    gemm2_weights_fp4 = gemm2_weights_quant.view(torch.float8_e4m3fn).reshape(
        num_experts, padded_hidden_size, padded_intermediate_size // 2
    )
    gemm2_scales_linear_fp4 = gemm2_scales_linear.view(torch.float8_e4m3fn).reshape(
        num_experts, padded_hidden_size, padded_intermediate_size // 32
    )

    # Shuffle weights and scales for each expert
    gemm1_weights_fp4_shuffled = []
    gemm1_scales_fp4_shuffled = []
    gemm1_bias_shuffled_list = []
    gemm2_weights_fp4_shuffled = []
    gemm2_scales_fp4_shuffled = []
    gemm2_bias_shuffled_list = []

    # Number of columns in the weight matrices (for computing row permutation)
    gemm1_ncols = padded_hidden_size // 2
    gemm2_ncols = padded_intermediate_size // 2

    for i in range(num_experts):
        # Shuffle gemm1 weights
        permute_indices = _maybe_get_cached_w3_w1_permute_indices(
            cache_permute_indices,
            gemm1_weights_fp4[i].view(torch.uint8),
            epilogue_tile_m,
        )
        gemm1_weights_fp4_shuffled.append(
            gemm1_weights_fp4[i]
            .view(torch.uint8)[permute_indices.to(gemm1_weights_fp4.device)]
            .contiguous()
        )

        # Shuffle gemm1 bias using row permutation derived from weight permutation
        if gate_up_bias is not None:
            gemm1_bias_shuffled_list.append(
                    gate_up_bias[i][permute_indices.to(gate_up_bias.device)]
                    .contiguous()
                )

        # Shuffle gemm1 scales
        permute_sf_indices = _maybe_get_cached_w3_w1_permute_indices(
            cache_permute_indices,
            gemm1_scales_linear_fp4[i].view(torch.uint8),
            epilogue_tile_m,
            num_elts_per_sf=16,
        )
        gemm1_scales_fp4_shuffled.append(
            block_scale_interleave(
                gemm1_scales_linear_fp4[i]
                .view(torch.uint8)[
                    permute_sf_indices.to(gemm1_scales_linear_fp4.device)
                ]
                .contiguous()
            )
        )

        # Shuffle gemm2 weights
        permute_indices = get_w2_permute_indices_with_cache(
            cache_permute_indices,
            gemm2_weights_fp4[i].view(torch.uint8),
            epilogue_tile_m,
        )
        gemm2_weights_fp4_shuffled.append(
            gemm2_weights_fp4[i]
            .view(torch.uint8)[permute_indices.to(gemm2_weights_fp4.device)]
            .contiguous()
        )

        # Shuffle gemm2 bias using row permutation derived from weight permutation
        if down_bias is not None:
            gemm2_bias_shuffled_list.append(
                down_bias[i][permute_indices.to(down_bias.device)]
                .contiguous()
            )

        # Shuffle gemm2 scales
        permute_sf_indices = get_w2_permute_indices_with_cache(
            cache_permute_indices,
            gemm2_scales_linear_fp4[i].view(torch.uint8),
            epilogue_tile_m,
            num_elts_per_sf=16,
        )
        gemm2_scales_fp4_shuffled.append(
            block_scale_interleave(
                gemm2_scales_linear_fp4[i]
                .view(torch.uint8)[
                    permute_sf_indices.to(gemm2_scales_linear_fp4.device)
                ]
                .contiguous()
            )
        )

    # Stack weights for all experts
    gemm1_weights_shuffled = torch.stack(gemm1_weights_fp4_shuffled)
    gemm1_scales_shuffled = (
        torch.stack(gemm1_scales_fp4_shuffled)
        .view(torch.float8_e4m3fn)
        .reshape(num_experts, 2 * padded_intermediate_size, padded_hidden_size // 32)
    )

    gemm2_weights_shuffled = torch.stack(gemm2_weights_fp4_shuffled)
    gemm2_scales_shuffled = (
        torch.stack(gemm2_scales_fp4_shuffled)
        .view(torch.float8_e4m3fn)
        .reshape(num_experts, padded_hidden_size, padded_intermediate_size // 32)
    )

    # Stack bias tensors if provided
    gemm1_bias_shuffled = None
    gemm2_bias_shuffled = None
    if gate_up_bias is not None:
        gemm1_bias_shuffled = torch.stack(gemm1_bias_shuffled_list)
    if down_bias is not None:
        gemm2_bias_shuffled = torch.stack(gemm2_bias_shuffled_list)

    return (
        gemm1_weights_shuffled,
        gemm1_scales_shuffled,
        gemm2_weights_shuffled,
        gemm2_scales_shuffled,
        gemm1_bias_shuffled,
        gemm2_bias_shuffled,
    )


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
            target_gate_up = f"{name}.gate_up_proj"
            blocks_gate_up = f"{name}.gate_up_proj_blocks"
            scales_gate_up = f"{name}.gate_up_proj_scales"
            fusion_map[target_gate_up] = {
                "sources": [blocks_gate_up, scales_gate_up],
                "op": "dequantize_mxfp4",
                "fp4_values": FP4_VALUES,
            }

            # Handle down_proj weights (MXFP4 format)
            target_down = f"{name}.down_proj"
            blocks_down = f"{name}.down_proj_blocks"
            scales_down = f"{name}.down_proj_scales"
            fusion_map[target_down] = {
                "sources": [blocks_down, scales_down],
                "op": "dequantize_mxfp4",
                "fp4_values": FP4_VALUES,
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

    def get_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Get raw router logits without top-k selection."""
        hidden_states = hidden_states.reshape(-1, self.hidden_size)
        router_logits = torch.nn.functional.linear(  # pylint: disable=not-callable
            hidden_states, self.weight, self.bias
        )
        return router_logits

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the router."""
        router_logits = self.get_logits(hidden_states)

        router_top_value, router_indices = torch.topk(
            router_logits, self.experts_per_token, dim=-1, sorted=True
        )

        router_top_value = torch.nn.functional.softmax(router_top_value, dim=1)
        router_scores = torch.zeros_like(router_logits).scatter_(
            1, router_indices, router_top_value
        )
        return router_scores, router_indices


class GptOssExperts(nn.Module):
    """GPT OSS Experts layer using FlashInfer fused MoE with MXFP4 weights."""

    def __init__(self, config: GptOssArch):
        """Initialize the GPT OSS Experts layer."""
        super().__init__()
        self.config = config
        self.num_experts = config.num_experts
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.swiglu_limit = config.swiglu_limit
        self.experts_per_token = config.experts_per_token

        # Compute padded dimensions for FlashInfer alignment requirements
        self.padded_hidden_size = pad_to_multiple(config.hidden_size, ALIGNMENT)
        self.padded_intermediate_size = pad_to_multiple(
            config.intermediate_size, ALIGNMENT
        )

        # Original dequantized weights (will be used to prepare kernel weights)
        self.gate_up_proj = nn.Parameter(
            torch.empty(
                (
                    config.num_experts,
                    config.intermediate_size * 2,
                    config.hidden_size,
                ),
                device=torch.device(config.device),
                dtype=config.dtype,
            )
        )
        self.gate_up_proj_bias = nn.Parameter(
            torch.empty(
                (config.num_experts, config.intermediate_size * 2),
                device=torch.device(config.device),
                dtype=config.dtype,
            )
        )
        self.down_proj = nn.Parameter(
            torch.empty(
                (
                    config.num_experts,
                    config.hidden_size,
                    config.intermediate_size,
                ),
                device=torch.device(config.device),
                dtype=config.dtype,
            )
        )
        self.down_proj_bias = nn.Parameter(
            torch.empty(
                (config.num_experts, config.hidden_size),
                device=torch.device(config.device),
                dtype=config.dtype,
            )
        )

        # Prepared weights for FlashInfer kernel (lazily initialized)
        self._weights_prepared = False
        self._cache_permute_indices: Dict[tuple, torch.Tensor] = {}
        self._gemm1_weights_shuffled: torch.Tensor | None = None
        self._gemm1_scales_shuffled: torch.Tensor | None = None
        self._gemm2_weights_shuffled: torch.Tensor | None = None
        self._gemm2_scales_shuffled: torch.Tensor | None = None
        self._gemm1_bias_shuffled: torch.Tensor | None = None
        self._gemm2_bias_shuffled: torch.Tensor | None = None
        self._gemm1_alpha: torch.Tensor | None = None
        self._gemm1_beta: torch.Tensor | None = None
        self._gemm1_clamp_limit: torch.Tensor | None = None

    def _prepare_weights(self):
        """
        Prepare weights for FlashInfer fused MoE kernel.
        
        This includes:
        1. De-interleaving gate_up weights from GPT OSS format to FlashInfer format
        2. Padding weights and biases to multiples of 256
        3. Re-quantizing to MXFP4 with proper shuffling
        """
        if self._weights_prepared:
            return

        # Step 1: De-interleave gate_up weights and bias
        # GPT OSS: interleaved [gate, linear, gate, linear, ...]
        # FlashInfer: non-interleaved [linear..., gate...]
        gate_up_deinterleaved = deinterleave_gate_up_weights(self.gate_up_proj.data)
        gate_up_bias_deinterleaved = deinterleave_gate_up_bias(
            self.gate_up_proj_bias.data
        )

        # Step 2: Pad weights to multiples of 256
        gate_up_padded = pad_gate_up_weights(
            gate_up_deinterleaved,
            self.padded_hidden_size,
            self.padded_intermediate_size,
        )
        down_padded = pad_down_weights(
            self.down_proj.data,
            self.padded_hidden_size,
            self.padded_intermediate_size,
        )

        # Step 3: Pad biases
        # FlashInfer expects bias in float32
        gemm1_bias_padded = pad_gate_up_bias(
            gate_up_bias_deinterleaved, self.padded_intermediate_size
        ).to(torch.float32)
        gemm2_bias_padded = pad_down_bias(
            self.down_proj_bias.data, self.padded_hidden_size
        ).to(torch.float32)

        # Step 4: Quantize and shuffle weights for kernel (includes bias shuffling)
        (
            self._gemm1_weights_shuffled,
            self._gemm1_scales_shuffled,
            self._gemm2_weights_shuffled,
            self._gemm2_scales_shuffled,
            self._gemm1_bias_shuffled,
            self._gemm2_bias_shuffled,
        ) = prepare_moe_weights_for_kernel(
            gate_up_padded,
            down_padded,
            self.padded_hidden_size,
            self.padded_intermediate_size,
            self.num_experts,
            self._cache_permute_indices,
            gate_up_bias=gemm1_bias_padded,
            down_bias=gemm2_bias_padded,
        )

        # Step 5: Prepare activation parameters for GPT OSS style SwiGLU
        # Alpha = 1.702 for sigmoid scaling
        self._gemm1_alpha = torch.full(
            (self.num_experts,),
            1.702,
            device=self.gate_up_proj.device,
            dtype=torch.float32,
        )
        # Beta = 1.0 for linear offset (x_linear + 1)
        self._gemm1_beta = torch.full(
            (self.num_experts,),
            1.0,
            device=self.gate_up_proj.device,
            dtype=torch.float32,
        )
        # Clamp limit for activation inputs
        self._gemm1_clamp_limit = torch.full(
            (self.num_experts,),
            self.swiglu_limit,
            device=self.gate_up_proj.device,
            dtype=torch.float32,
        )

        self._weights_prepared = True

    def old_forward(self, t: torch.Tensor, expert_indices: torch.Tensor) -> torch.Tensor:
        """Old forward pass through the experts (manual MoE implementation)."""

        # Gate and Up projection
        gate_up_proj = self.gate_up_proj[expert_indices, ...]
        gate_up_proj_bias = self.gate_up_proj_bias[expert_indices, ...]

        t = torch.einsum("beck,bk->bec", gate_up_proj, t) + gate_up_proj_bias

        # Inline swiglu function
        x_glu, x_linear = t[..., ::2], t[..., 1::2]

        # Clamp the input values
        x_glu = x_glu.clamp(min=None, max=self.swiglu_limit)
        x_linear = x_linear.clamp(min=-self.swiglu_limit, max=self.swiglu_limit)
        out_glu = x_glu * torch.sigmoid(1.702 * x_glu)

        # Add an extra bias of 1 to the linear layer
        t = out_glu * (x_linear + 1)

        # Down projection
        down_proj = self.down_proj[expert_indices, ...]
        down_proj_bias = self.down_proj_bias[expert_indices, ...]

        t = torch.einsum("beck,bek->bec", down_proj, t) + down_proj_bias

        return t

    def reference_forward(
        self, hidden_states: torch.Tensor, expert_logits: torch.Tensor
    ) -> torch.Tensor:
        """
        Reference forward pass using the test script's reference implementation.
        
        This helps verify the FlashInfer kernel behavior.
        """
        num_tokens = hidden_states.shape[0]
        padding = 8  # Same as test script

        # De-interleave gate_up weights from GPT OSS format to FlashInfer format
        gate_up_deinterleaved = deinterleave_gate_up_weights(self.gate_up_proj.data)
        gate_up_bias_deinterleaved = deinterleave_gate_up_bias(
            self.gate_up_proj_bias.data
        )

        # Get permute info using renormalize routing (TopK -> Softmax)
        permute_info, _ = routing_reference_renormalize(
            expert_logits, self.experts_per_token, self.num_experts, padding
        )

        # Prepare activation parameters
        gemm1_alpha = torch.full(
            (self.num_experts,), 1.702, device=hidden_states.device, dtype=torch.float32
        )
        gemm1_beta = torch.full(
            (self.num_experts,), 1.0, device=hidden_states.device, dtype=torch.float32
        )
        gemm1_clamp_limit = torch.full(
            (self.num_experts,),
            self.swiglu_limit,
            device=hidden_states.device,
            dtype=torch.float32,
        )

        # Run reference MoE
        output = run_moe_reference(
            num_tokens=num_tokens,
            num_experts=self.num_experts,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            top_k=self.experts_per_token,
            padding=padding,
            hidden_states=hidden_states.float(),
            gemm1_weights=gate_up_deinterleaved.float(),
            gemm2_weights=self.down_proj.data.float(),
            permute_info=permute_info,
            gemm1_bias=gate_up_bias_deinterleaved.float(),
            gemm2_bias=self.down_proj_bias.data.float(),
            gemm1_alpha=gemm1_alpha,
            gemm1_beta=gemm1_beta,
            gemm1_clamp_limit=gemm1_clamp_limit,
        )

        return output.to(hidden_states.dtype)

    def forward(
        self, hidden_states: torch.Tensor, expert_logits: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through the experts using FlashInfer fused MoE.
        
        Args:
            hidden_states: Input tensor of shape [num_tokens, hidden_size]
            expert_logits: Router logits of shape [num_tokens, num_experts]
        
        Returns:
            Output tensor of shape [num_tokens, hidden_size]
        """
        # Lazily prepare weights on first forward pass
        self._prepare_weights()

        # Assert weights are prepared (for type checker)
        assert self._gemm1_weights_shuffled is not None
        assert self._gemm1_scales_shuffled is not None
        assert self._gemm2_weights_shuffled is not None
        assert self._gemm2_scales_shuffled is not None

        # Ensure hidden states are bfloat16
        hidden_states_bf16 = hidden_states.to(torch.bfloat16)

        # Pad hidden states if necessary
        if self.hidden_size != self.padded_hidden_size:
            num_tokens = hidden_states_bf16.shape[0]
            padded_hidden = torch.zeros(
                (num_tokens, self.padded_hidden_size),
                dtype=hidden_states_bf16.dtype,
                device=hidden_states_bf16.device,
            )
            padded_hidden[:, : self.hidden_size] = hidden_states_bf16
            hidden_states_bf16 = padded_hidden

        # Call FlashInfer fused MoE kernel
        output = trtllm_fp4_block_scale_moe(
            routing_logits=expert_logits.to(torch.bfloat16),
            routing_bias=None,
            hidden_states=hidden_states_bf16,
            hidden_states_scale=None,  # BF16 doesn't need scale
            gemm1_weights=self._gemm1_weights_shuffled,
            gemm1_weights_scale=self._gemm1_scales_shuffled,
            gemm1_bias=self._gemm1_bias_shuffled,
            gemm1_alpha=self._gemm1_alpha,
            gemm1_beta=self._gemm1_beta,
            gemm1_clamp_limit=self._gemm1_clamp_limit,
            gemm2_weights=self._gemm2_weights_shuffled,
            gemm2_weights_scale=self._gemm2_scales_shuffled,
            gemm2_bias=self._gemm2_bias_shuffled,
            output1_scale_scalar=torch.full(
                (self.num_experts,), 1.0, device=hidden_states.device
            ),
            output1_scale_gate_scalar=torch.full(
                (self.num_experts,), 1.0, device=hidden_states.device
            ),
            output2_scale_scalar=torch.full(
                (self.num_experts,), 1.0, device=hidden_states.device
            ),
            num_experts=self.num_experts,
            top_k=self.experts_per_token,
            n_group=None,
            topk_group=None,
            intermediate_size=self.padded_intermediate_size,
            local_expert_offset=0,
            local_num_experts=self.num_experts,
            routed_scaling_factor=None,
            tile_tokens_dim=None,
            routing_method_type=1,  # 1: Renormalize (TopK -> Softmax)
            gated_act_type=0,  # 0: SwiGlu
            do_finalize=True,
            tune_max_num_tokens=TUNE_MAX_NUM_TOKENS,
        )

        # Handle different return types
        if isinstance(output, (tuple, list)):
            output = output[0]

        # Strip padding from output
        if self.hidden_size != self.padded_hidden_size:
            output = output[:, : self.hidden_size]

        return output.to(hidden_states.dtype)


class GptOssMlp(nn.Module):
    """GPT OSS MLP layer with Mixture of Experts using FlashInfer fused MoE."""

    def __init__(self, config: GptOssArch):
        """Initialize the GPT OSS MLP layer."""
        super().__init__()
        self.config = config
        self.router = GptOssRouter(config)
        self.experts = GptOssExperts(config)

    def old_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Old forward pass through the MLP layer (manual MoE implementation)."""
        # Router determines expert selection and weights
        router_scores, router_indices = self.router(x)

        # Extract the weights for selected experts
        expert_weights = torch.gather(router_scores, 1, router_indices)

        # Forward through experts
        t = self.experts.old_forward(x, router_indices)

        # Weighted sum of experts
        t = torch.einsum("bec,be->bc", t, expert_weights)

        return t

    def reference_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Reference forward using the test script's reference implementation."""
        router_logits = self.router.get_logits(x)
        return self.experts.reference_forward(x, router_logits)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the MLP layer."""
        # Get router logits (the fused MoE kernel handles routing internally)
        router_logits = self.router.get_logits(x)

        # Forward through experts with fused MoE
        # The FlashInfer kernel handles top-k selection, softmax, and weighted sum
        output = self.experts(x, router_logits)

        return output


class GptOssDecoderLayer(nn.Module):
    """GPT OSS decoder layer."""

    def __init__(self, config: GptOssArch, layer_idx: int):
        """Initialize the GPT OSS decoder layer."""
        super().__init__()
        self.layer_idx = layer_idx
        self._first_forward_done = False  # For debugging MoE implementations
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

        # Debug comparison: only on first layer, first forward call
        if self.layer_idx == 0 and not self._first_forward_done:
            self._first_forward_done = True
            print(f"\n[DEBUG] Layer {self.layer_idx}: Comparing MoE implementations...")
            print(f"[DEBUG] Input shape: {hidden_states.shape}")

            # Run all three implementations
            out_old = self.mlp.old_forward(hidden_states)
            out_ref = self.mlp.reference_forward(hidden_states)
            out_new = self.mlp(hidden_states)

            # Compare old vs reference
            old_ref_diff = (out_old.float() - out_ref.float()).abs()
            old_ref_max = old_ref_diff.max().item()
            old_ref_mean = old_ref_diff.mean().item()
            old_ref_close = torch.isclose(
                out_old.float(), out_ref.float(), atol=0.1, rtol=0.1
            )
            old_ref_match = old_ref_close.float().mean().item()

            print(f"[DEBUG] old_forward vs reference_forward:")
            print(f"[DEBUG]   max diff: {old_ref_max:.6f}")
            print(f"[DEBUG]   mean diff: {old_ref_mean:.6f}")
            print(f"[DEBUG]   match ratio (atol=0.1, rtol=0.1): {old_ref_match:.4f}")

            # Compare old vs FlashInfer
            old_new_diff = (out_old.float() - out_new.float()).abs()
            old_new_max = old_new_diff.max().item()
            old_new_mean = old_new_diff.mean().item()
            old_new_close = torch.isclose(
                out_old.float(), out_new.float(), atol=0.1, rtol=0.1
            )
            old_new_match = old_new_close.float().mean().item()

            print(f"[DEBUG] old_forward vs FlashInfer forward:")
            print(f"[DEBUG]   max diff: {old_new_max:.6f}")
            print(f"[DEBUG]   mean diff: {old_new_mean:.6f}")
            print(f"[DEBUG]   match ratio (atol=0.1, rtol=0.1): {old_new_match:.4f}")

            # Compare reference vs FlashInfer
            ref_new_diff = (out_ref.float() - out_new.float()).abs()
            ref_new_max = ref_new_diff.max().item()
            ref_new_mean = ref_new_diff.mean().item()
            ref_new_close = torch.isclose(
                out_ref.float(), out_new.float(), atol=0.1, rtol=0.1
            )
            ref_new_match = ref_new_close.float().mean().item()

            print(f"[DEBUG] reference_forward vs FlashInfer forward:")
            print(f"[DEBUG]   max diff: {ref_new_max:.6f}")
            print(f"[DEBUG]   mean diff: {ref_new_mean:.6f}")
            print(f"[DEBUG]   match ratio (atol=0.1, rtol=0.1): {ref_new_match:.4f}")

            # Print some sample values
            print(f"[DEBUG] Sample outputs (first 5 elements of first token):")
            print(f"[DEBUG]   old_forward: {out_old[0, :5].tolist()}")
            print(f"[DEBUG]   reference:   {out_ref[0, :5].tolist()}")
            print(f"[DEBUG]   FlashInfer:  {out_new[0, :5].tolist()}")
            print()

            # If FlashInfer output differs significantly, dump weights for debugging
            if old_new_match < 0.9:
                dump_dir = "/tmp/moe_debug"
                import os
                os.makedirs(dump_dir, exist_ok=True)
                print(f"[DEBUG] FlashInfer match ratio too low, dumping weights to {dump_dir}")

                # Get router logits for the dump
                router_logits = self.mlp.router.get_logits(hidden_states)

                # Dump inputs and weights
                torch.save(hidden_states, f"{dump_dir}/hidden_states.pt")
                torch.save(router_logits, f"{dump_dir}/router_logits.pt")
                torch.save(
                    self.mlp.experts.gate_up_proj.data, f"{dump_dir}/gate_up_proj.pt"
                )
                torch.save(
                    self.mlp.experts.gate_up_proj_bias.data,
                    f"{dump_dir}/gate_up_proj_bias.pt",
                )
                torch.save(
                    self.mlp.experts.down_proj.data, f"{dump_dir}/down_proj.pt"
                )
                torch.save(
                    self.mlp.experts.down_proj_bias.data,
                    f"{dump_dir}/down_proj_bias.pt",
                )

                # Save model config info
                config_info = {
                    "num_experts": self.mlp.experts.num_experts,
                    "hidden_size": self.mlp.experts.hidden_size,
                    "intermediate_size": self.mlp.experts.intermediate_size,
                    "experts_per_token": self.mlp.experts.experts_per_token,
                    "swiglu_limit": self.mlp.experts.swiglu_limit,
                }
                torch.save(config_info, f"{dump_dir}/config_info.pt")

                print(f"[DEBUG] Dumped: hidden_states, router_logits, weights, config")
                print(f"[DEBUG] Config: {config_info}")

                exit(1)

            # Use old_forward output for now (known working)
            hidden_states = out_old
        else:
            # Use FlashInfer for subsequent calls
            hidden_states = self.mlp.old_forward(hidden_states)

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
