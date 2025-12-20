"""
Sanity check script for FlashInfer fused MoE with MXFP4 weights and BF16 activation.

This script tests our understanding of the FlashInfer `trtllm_fp4_block_scale_moe` function
by comparing its output against a reference dequantized implementation.

Based on the official FlashInfer tests:
https://github.com/flashinfer-ai/flashinfer/blob/0e68a2febc58df99429f652769c5c485ca67fc39/tests/moe/test_trtllm_gen_fused_moe.py
"""

from __future__ import annotations

from typing import Dict

import pytest
import torch
from torch.nn import functional as F

from flashinfer import (
    e2m1_and_ufp8sf_scale_to_float,
    fp4_quantize,
)
from flashinfer.autotuner import autotune
from flashinfer.fp4_quantization import block_scale_interleave
from flashinfer.fused_moe import (
    WeightLayout,
    trtllm_fp4_block_scale_moe,
)
from flashinfer.fused_moe.core import (
    get_w2_permute_indices_with_cache,
    _maybe_get_cached_w3_w1_permute_indices,
)

# Max num tokens to tune for trtllm-gen fused moe
TUNE_MAX_NUM_TOKENS = 4096

# Alignment requirement for trtllm_fp4_block_scale_moe
ALIGNMENT = 256


# ====================================================================================
# Padding Functions for trtllm_fp4_block_scale_moe
# ====================================================================================


def pad_to_multiple(size: int, multiple: int = ALIGNMENT) -> int:
    """Calculate padded size to be a multiple of the given number."""
    if size % multiple == 0:
        return size
    return ((size + multiple - 1) // multiple) * multiple


def pad_hidden_states(
    hidden_states: torch.Tensor,
    padded_hidden_size: int,
) -> torch.Tensor:
    """
    Pad hidden states from [num_tokens, hidden_size] to [num_tokens, padded_hidden_size].
    
    Args:
        hidden_states: Input tensor of shape [num_tokens, hidden_size]
        padded_hidden_size: Target hidden size (must be >= original hidden_size)
    
    Returns:
        Padded tensor of shape [num_tokens, padded_hidden_size]
    """
    num_tokens, hidden_size = hidden_states.shape
    if hidden_size == padded_hidden_size:
        return hidden_states
    
    padded = torch.zeros(
        (num_tokens, padded_hidden_size),
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    padded[:, :hidden_size] = hidden_states
    return padded


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
    
    Args:
        weights: Input tensor of shape [num_experts, 2*intermediate_size, hidden_size]
        padded_hidden_size: Target hidden size
        padded_intermediate_size: Target intermediate size
    
    Returns:
        Padded tensor of shape [num_experts, 2*padded_intermediate_size, padded_hidden_size]
    """
    num_experts, fused_size, hidden_size = weights.shape
    intermediate_size = fused_size // 2
    
    if hidden_size == padded_hidden_size and intermediate_size == padded_intermediate_size:
        return weights
    
    # Split into linear and gate parts
    linear_part = weights[:, :intermediate_size, :]  # [num_experts, intermediate_size, hidden_size]
    gate_part = weights[:, intermediate_size:, :]    # [num_experts, intermediate_size, hidden_size]
    
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
    
    Args:
        weights: Input tensor of shape [num_experts, hidden_size, intermediate_size]
        padded_hidden_size: Target hidden size
        padded_intermediate_size: Target intermediate size
    
    Returns:
        Padded tensor of shape [num_experts, padded_hidden_size, padded_intermediate_size]
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
    
    Args:
        bias: Input tensor of shape [num_experts, 2*intermediate_size]
        padded_intermediate_size: Target intermediate size
    
    Returns:
        Padded tensor of shape [num_experts, 2*padded_intermediate_size]
    """
    num_experts, fused_size = bias.shape
    intermediate_size = fused_size // 2
    
    if intermediate_size == padded_intermediate_size:
        return bias
    
    # Split into linear and gate parts
    linear_part = bias[:, :intermediate_size]  # [num_experts, intermediate_size]
    gate_part = bias[:, intermediate_size:]    # [num_experts, intermediate_size]
    
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
    
    Args:
        bias: Input tensor of shape [num_experts, hidden_size]
        padded_hidden_size: Target hidden size
    
    Returns:
        Padded tensor of shape [num_experts, padded_hidden_size]
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


# ====================================================================================
# FP4 Quantization Functions
# ====================================================================================


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
        # The test case uses MXFP4, so global scale factor is 1.0
        a_fp4, a_sf = fp4_quantize(
            a[i].cuda(), a_global_sf, sf_vec_size, True, is_sf_swizzled_layout
        )
        quant_a.append(a_fp4)
        sfs.append(a_sf)

    result_quant_a = torch.stack(quant_a)
    result_sfs = torch.stack(sfs)

    return result_quant_a, result_sfs


def dequant_mxfp4_batches(
    mat_fp4: torch.Tensor,
    scale_tensor: torch.Tensor,
    global_scale_tensor: torch.Tensor,
    is_sf_swizzled_layout: bool,
):
    """Batch FP4 dequantization helper."""
    num_experts = mat_fp4.shape[0]
    sf_vec_size = 32
    ufp8_type = 0

    scale_tensor = scale_tensor.view(num_experts, -1)

    tensors = [
        e2m1_and_ufp8sf_scale_to_float(
            mat_fp4[b, :, :].cpu(),
            scale_tensor[b, :].cpu().reshape(-1),
            global_scale_tensor[b].cpu(),
            sf_vec_size,
            ufp8_type,
            is_sf_swizzled_layout,
        )
        for b in range(num_experts)
    ]

    result = torch.stack(tensors)
    return result


# ====================================================================================
# Routing Reference Implementations
# ====================================================================================


def routing_reference(expertLogits, topK, padding):
    """Reference routing implementation for permutation calculation."""
    originalDevice = expertLogits.device
    expertLogits = expertLogits.cpu()
    numTokens, numExperts = expertLogits.shape
    assert topK <= numExperts

    numTokensPerExpert = torch.zeros(numExperts, dtype=torch.int64)
    expandedTokenIdxToExpert = -torch.ones(numTokens * topK, dtype=torch.int64)
    expandedTokenIdxToIdxInExpert = -torch.ones(numTokens * topK, dtype=torch.int64)

    topKLogits, topKIndices = torch.topk(expertLogits, topK, dim=1)
    for tokenIdx in range(numTokens):
        for k in range(topK):
            expandedIdx = tokenIdx * topK + k
            expertIndex = topKIndices[tokenIdx, k]
            expandedTokenIdxToExpert[expandedIdx] = expertIndex
            expandedTokenIdxToIdxInExpert[expandedIdx] = numTokensPerExpert[expertIndex]
            numTokensPerExpert[expertIndex] += 1

    paddedTokensPerExpertPrefixSum = torch.zeros(numExperts + 1, dtype=torch.int64)
    for ii in range(numExperts):

        def divUpMul(a, b):
            return (a + b - 1) // b * b

        paddedTokensPerExpertPrefixSum[ii + 1] = paddedTokensPerExpertPrefixSum[
            ii
        ] + divUpMul(numTokensPerExpert[ii], padding)
    permutedBufferSize = paddedTokensPerExpertPrefixSum[numExperts]

    expandedTokenIdxToPermutedIdx = -torch.ones(numTokens * topK, dtype=torch.int64)
    permutedIdxToExpandedIdx = -torch.ones(permutedBufferSize, dtype=torch.int64)
    permutedIdxToTokenIdx = -torch.ones(permutedBufferSize, dtype=torch.int64)
    for tokenIdx in range(numTokens):
        for k in range(topK):
            expandedIdx = tokenIdx * topK + k
            expert = expandedTokenIdxToExpert[expandedIdx]
            offsetWithinExpert = expandedTokenIdxToIdxInExpert[expandedIdx]
            offsetForExpert = paddedTokensPerExpertPrefixSum[expert]
            permutedIdx = offsetForExpert + offsetWithinExpert

            expandedTokenIdxToPermutedIdx[expandedIdx] = permutedIdx
            permutedIdxToExpandedIdx[permutedIdx] = expandedIdx
            permutedIdxToTokenIdx[permutedIdx] = tokenIdx
    return {
        "paddedTokensPerExpertPrefixSum": paddedTokensPerExpertPrefixSum.to(
            originalDevice
        ),
        "permutedBufferSize": permutedBufferSize.item(),
        "expandedTokenIdxToPermutedIdx": expandedTokenIdxToPermutedIdx.to(
            originalDevice
        ),
        "permutedIdxToExpandedIdx": permutedIdxToExpandedIdx.to(originalDevice),
        "numTokensPerExpert": numTokensPerExpert.to(originalDevice),
        "expandedTokenIdxToExpert": expandedTokenIdxToExpert.to(originalDevice),
        "topKLogits": topKLogits.to(originalDevice),
        "permutedIdxToTokenIdx": permutedIdxToTokenIdx.to(originalDevice),
        "topKIndices": topKIndices.to(originalDevice),
    }


def routing_reference_renormalize(expert_logits, top_k, num_experts, padding):
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


# ====================================================================================
# Common MoE Reference Implementation
# ====================================================================================


def run_moe_dequant(
    num_tokens,
    num_experts,
    hidden_size,
    intermediate_size,
    top_k,
    padding,
    hidden_states,
    gemm1_weights,
    gemm2_weights,
    permute_info,
    gemm1_bias=None,
    gemm2_bias=None,
    gemm1_alpha=None,
    gemm1_beta=None,
    gemm1_clamp_limit=None,
):
    """Common dequantized MoE reference implementation."""
    # Permute
    total_num_padded_tokens = permute_info["permutedBufferSize"]
    expanded_idx_to_permuted_idx = permute_info[
        "expandedTokenIdxToPermutedIdx"
    ].cpu()
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
        # Add gemm1 bias if provided
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
        # x_glu (gated) clamped to max=limit
        # x_linear clamped to [-limit, limit]
        if gemm1_clamp_limit is not None:
            clamp_limit = gemm1_clamp_limit[expert_idx]
            my_x2 = my_x2.clamp(max=clamp_limit)
            my_x1 = my_x1.clamp(min=-clamp_limit, max=clamp_limit)

        # Apply activation with optional alpha scaling
        # Standard SwiGLU: silu(x2) * x1 = (x2 * sigmoid(x2)) * x1
        # With alpha: (x2 * sigmoid(alpha * x2)) * x1
        if gemm1_alpha is not None:
            alpha = gemm1_alpha[expert_idx]
            gated_output = my_x2 * torch.sigmoid(alpha * my_x2)
        else:
            gated_output = F.silu(my_x2)

        # Apply beta offset to linear part (GPT OSS style: out_glu * (x_linear + beta))
        if gemm1_beta is not None:
            beta = gemm1_beta[expert_idx]
            linear_output = my_x1 + beta
        else:
            linear_output = my_x1

        activation_output[i : i + my_num_tokens] = gated_output * linear_output
        i += my_num_tokens
        i = (i + padding - 1) // padding * padding

    # For MxFP4xBf16, just convert activation to bf16 and back
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
        # Add gemm2 bias if provided
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
            weight = (
                expert_weight[i, top_k_idx]
            )
            acc += original_vector * weight
        finalize_output[i] = acc
    return finalize_output


def run_moe_reference_fp4_bf16(
    num_tokens,
    num_experts,
    hidden_size,
    intermediate_size,
    top_k,
    padding,
    hidden_states,
    gemm1_weights,
    gemm1_scales,
    gemm2_weights,
    gemm2_scales,
    permute_info,
    gemm1_bias=None,
    gemm2_bias=None,
    gemm1_alpha=None,
    gemm1_beta=None,
    gemm1_clamp_limit=None,
):
    """FP4 with BF16 activation reference implementation."""

    # Hidden states are already BF16, just convert to float for reference
    hidden_states_dequant = hidden_states.to(torch.bfloat16).to(torch.float)

    gemm1_weights_dequant = dequant_mxfp4_batches(
        gemm1_weights,
        gemm1_scales,
        torch.full((num_experts,), 1.0, device="cuda"),
        is_sf_swizzled_layout=True,
    ).cuda()

    gemm2_weights_dequant = dequant_mxfp4_batches(
        gemm2_weights,
        gemm2_scales,
        torch.full((num_experts,), 1.0, device="cuda"),
        is_sf_swizzled_layout=True,
    ).cuda()

    return run_moe_dequant(
        num_tokens,
        num_experts,
        hidden_size,
        intermediate_size,
        top_k,
        padding,
        hidden_states_dequant,
        gemm1_weights_dequant,
        gemm2_weights_dequant,
        permute_info,
        gemm1_bias=gemm1_bias,
        gemm2_bias=gemm2_bias,
        gemm1_alpha=gemm1_alpha,
        gemm1_beta=gemm1_beta,
        gemm1_clamp_limit=gemm1_clamp_limit,
    )


# ====================================================================================
# FP4Moe Implementation for MxFP4 x Bf16
# ====================================================================================


class FP4Moe:
    """
    FP4 MxFP4 MoE implementation with block scaling and BF16 activation.
    """

    def __init__(self):
        self._cache_permute_indices: Dict[tuple, torch.Tensor] = {}

    def quantize_weights(self, gemm1_weights, gemm2_weights):
        """Quantize weights to FP4 format.
        
        Returns:
            Tuple of (gemm1_weights_fp4, gemm1_scales, gemm2_weights_fp4, gemm2_scales)
        """

        # Quantize the weights for FC1
        gemm1_weights_fp4_bytes, gemm1_scales_fp4_bytes = (
            quant_mxfp4_batches(gemm1_weights, True)
        )

        # Quantize the weights for FC2
        gemm2_weights_fp4_bytes, gemm2_scales_fp4_bytes = (
            quant_mxfp4_batches(gemm2_weights, True)
        )

        return (
            gemm1_weights_fp4_bytes,
            gemm1_scales_fp4_bytes,
            gemm2_weights_fp4_bytes,
            gemm2_scales_fp4_bytes,
        )

    def prepare_static_weights_for_kernel(
        self,
        gemm1_weights_quant_padded,
        gemm2_weights_quant_padded,
        gemm1_weights_padded,
        gemm2_weights_padded,
        padded_hidden_size,
        padded_intermediate_size,
        num_experts,
        gemm1_bias_padded=None,
        gemm2_bias_padded=None,
    ):
        """Prepare quantized weights for kernel (done offline with weights).
        
        Args:
            gemm1_weights_quant_padded: Quantized gate_up weights (already padded)
            gemm2_weights_quant_padded: Quantized down weights (already padded)
            gemm1_weights_padded: Original padded gate_up weights (for scale calculation)
            gemm2_weights_padded: Original padded down weights (for scale calculation)
            padded_hidden_size: Padded hidden size (must be multiple of 256)
            padded_intermediate_size: Padded intermediate size (must be multiple of 256)
            num_experts: Number of experts
            gemm1_bias_padded: Optional padded gate_up bias [num_experts, 2*padded_intermediate_size]
            gemm2_bias_padded: Optional padded down bias [num_experts, padded_hidden_size]
        
        Returns:
            Tuple of (gemm1_weights_shuffled, gemm1_scales_shuffled, 
                      gemm2_weights_shuffled, gemm2_scales_shuffled,
                      gemm1_bias_shuffled, gemm2_bias_shuffled)
        """
        epilogue_tile_m = 128  # FIXME: this depends on the kernel internals

        # Quantize weights with linear layout for kernels
        _, gemm1_scales_linear_fp4_bytes = quant_mxfp4_batches(
            gemm1_weights_padded, False
        )
        _, gemm2_scales_linear_fp4_bytes = quant_mxfp4_batches(
            gemm2_weights_padded, False
        )

        # Convert quantized weights to proper formats
        gemm1_weights_fp4 = gemm1_weights_quant_padded.view(torch.float8_e4m3fn).reshape(
            num_experts, 2 * padded_intermediate_size, padded_hidden_size // 2
        )  # packed fp4
        gemm1_scales_linear_fp4 = gemm1_scales_linear_fp4_bytes.view(
            torch.float8_e4m3fn
        ).reshape(
            num_experts, 2 * padded_intermediate_size, padded_hidden_size // 32
        )  # fp8 scaling factors

        gemm2_weights_fp4 = gemm2_weights_quant_padded.view(torch.float8_e4m3fn).reshape(
            num_experts, padded_hidden_size, padded_intermediate_size // 2
        )  # packed fp4
        gemm2_scales_linear_fp4 = gemm2_scales_linear_fp4_bytes.view(
            torch.float8_e4m3fn
        ).reshape(
            num_experts, padded_hidden_size, padded_intermediate_size // 32
        )  # fp8 scaling factors

        # Using cached permute index calculation can speed up weights preprocessing
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
            # Calculate the permute indices for the following:
            # 1. Reorder rows of W1 and scales for fused gated activation
            # 2. Shuffle weights and scaling factors for transposed mma output
            # for both w3_w1 and w2 weights and scale factors
            permute_indices = _maybe_get_cached_w3_w1_permute_indices(
                self._cache_permute_indices,
                gemm1_weights_fp4[i].view(torch.uint8),
                epilogue_tile_m,
            )
            gemm1_weights_fp4_shuffled.append(
                gemm1_weights_fp4[i]
                .view(torch.uint8)[permute_indices.to(gemm1_weights_fp4.device)]
                .contiguous()
            )
            
            # Shuffle gemm1 bias using row permutation derived from weight permutation
            if gemm1_bias_padded is not None:
                gemm1_bias_shuffled_list.append(
                    gemm1_bias_padded[i][permute_indices.to(gemm1_bias_padded.device)]
                    .contiguous()
                )

            permute_sf_indices = _maybe_get_cached_w3_w1_permute_indices(
                self._cache_permute_indices,
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

            permute_indices = get_w2_permute_indices_with_cache(
                self._cache_permute_indices,
                gemm2_weights_fp4[i].view(torch.uint8),
                epilogue_tile_m,
            )
            gemm2_weights_fp4_shuffled.append(
                gemm2_weights_fp4[i]
                .view(torch.uint8)[permute_indices.to(gemm2_weights_fp4.device)]
                .contiguous()
            )
            
            # Shuffle gemm2 bias using row permutation derived from weight permutation
            if gemm2_bias_padded is not None:
                gemm2_bias_shuffled_list.append(
                    gemm2_bias_padded[i][permute_indices.to(gemm2_bias_padded.device)]
                    .contiguous()
                )

            permute_sf_indices = get_w2_permute_indices_with_cache(
                self._cache_permute_indices,
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
        gemm1_weights_fp4_shuffled = torch.stack(gemm1_weights_fp4_shuffled)
        gemm1_scales_fp4_shuffled = (
            torch.stack(gemm1_scales_fp4_shuffled)
            .view(torch.float8_e4m3fn)
            .reshape(
                num_experts, 2 * padded_intermediate_size, padded_hidden_size // 32
            )
        )

        gemm2_weights_fp4_shuffled = torch.stack(gemm2_weights_fp4_shuffled)
        gemm2_scales_fp4_shuffled = (
            torch.stack(gemm2_scales_fp4_shuffled)
            .view(torch.float8_e4m3fn)
            .reshape(num_experts, padded_hidden_size, padded_intermediate_size // 32)
        )
        
        # Stack bias tensors if provided
        gemm1_bias_shuffled = None
        gemm2_bias_shuffled = None
        if gemm1_bias_padded is not None:
            gemm1_bias_shuffled = torch.stack(gemm1_bias_shuffled_list)
        if gemm2_bias_padded is not None:
            gemm2_bias_shuffled = torch.stack(gemm2_bias_shuffled_list)

        return (
            gemm1_weights_fp4_shuffled,
            gemm1_scales_fp4_shuffled,
            gemm2_weights_fp4_shuffled,
            gemm2_scales_fp4_shuffled,
            gemm1_bias_shuffled,
            gemm2_bias_shuffled,
        )

    def call_moe(
        self,
        gemm1_weights_shuffled,
        gemm1_scales_shuffled,
        gemm2_weights_shuffled,
        gemm2_scales_shuffled,
        hidden_states_orig,
        expert_logits,
        num_experts,
        top_k,
        intermediate_size,
        gemm1_bias=None,
        gemm2_bias=None,
        gemm1_alpha=None,
        gemm1_beta=None,
        gemm1_clamp_limit=None,
    ):
        """Call the FlashInfer fused MoE kernel."""
        # BF16 hidden states
        hidden_states_bf16 = hidden_states_orig.to(torch.bfloat16)

        with autotune(True):
            output = trtllm_fp4_block_scale_moe(
                routing_logits=expert_logits,
                routing_bias=None,
                hidden_states=hidden_states_bf16,
                hidden_states_scale=None,  # BF16 doesn't need scale
                gemm1_weights=gemm1_weights_shuffled,
                gemm1_weights_scale=gemm1_scales_shuffled,
                gemm1_bias=gemm1_bias,
                gemm1_alpha=gemm1_alpha,
                gemm1_beta=gemm1_beta,
                gemm1_clamp_limit=gemm1_clamp_limit,
                gemm2_weights=gemm2_weights_shuffled,
                gemm2_weights_scale=gemm2_scales_shuffled,
                gemm2_bias=gemm2_bias,
                output1_scale_scalar=torch.full((num_experts,), 1.0, device="cuda"),
                output1_scale_gate_scalar=torch.full((num_experts,), 1.0, device="cuda"),
                output2_scale_scalar=torch.full((num_experts,), 1.0, device="cuda"),
                num_experts=num_experts,
                top_k=top_k,
                n_group=None,
                topk_group=None,
                intermediate_size=intermediate_size,
                local_expert_offset=0,
                local_num_experts=num_experts,
                routed_scaling_factor=None,
                tile_tokens_dim=None,
                routing_method_type=1, # 1: Renormalize (TopK -> Softmax)
                gated_act_type=0, # 0: SwiGlu 
                do_finalize=True,
                tune_max_num_tokens=TUNE_MAX_NUM_TOKENS,
            )
        # Handle different return types (tuple, list, or tensor)
        if isinstance(output, (tuple, list)):
            return output[0].to(torch.float)
        return output.to(torch.float)

    def get_tolerances(self):
        """Get FP4-specific accuracy tolerances."""
        return {"atol": 0.1, "rtol": 0.85, "percent": 0.925}


# ====================================================================================
# Accuracy Check
# ====================================================================================


def check_accuracy(a, b, atol, rtol, percent):
    """Unified accuracy checking function with detailed error reporting."""
    if not torch.isfinite(a).all():
        raise Exception("Non-finite values in reference output")
    if not torch.isfinite(b).all():
        raise Exception("Non-finite values in actual output")
    assert a.shape == b.shape, f"Shape mismatch: {a.shape} vs {b.shape}"

    close = torch.isclose(a, b, atol=atol, rtol=rtol)
    match_ratio = close.float().mean()
    if match_ratio >= percent:
        print(f"✓ Match ratio: {match_ratio:.4f} >= {percent:.4f}")
        return

    mismatch_percent = 1.0 - match_ratio.item()
    if mismatch_percent > 1 - percent:
        raise Exception(
            f"Mismatch percentage is {mismatch_percent:.4f} for rtol {rtol} "
            f"(threshold: {1 - percent:.4f})"
        )


# ====================================================================================
# Test Function
# ====================================================================================


def run_moe_test(
    num_tokens: int = 128,
    hidden_size: int = 1024,
    intermediate_size: int = 1024,
    num_experts: int = 128,
    top_k: int = 8,
    padding: int = 8,
    use_gemm_bias: bool = False,
    gemm1_alpha_value: float | None = None,
    gemm1_beta_value: float | None = None,
    gemm1_clamp_limit_value: float | None = None,
):
    """
    Test MxFP4 x BF16 fused MoE against reference implementation.

    This test handles padding for trtllm_fp4_block_scale_moe which requires
    hidden_size and intermediate_size to be multiples of 256.
    
    - Reference implementation uses original (unpadded) dimensions
    - FlashInfer kernel uses padded dimensions
    - Output is stripped of padding before comparison

    Args:
        num_tokens: Number of input tokens
        hidden_size: Hidden dimension size (will be padded to multiple of 256 for kernel)
        intermediate_size: Intermediate (FFN) dimension size (will be padded to multiple of 256 for kernel)
        num_experts: Number of experts in MoE
        top_k: Number of experts selected per token
        padding: Padding for expert token counts
        use_gemm_bias: Whether to use bias for GEMM1 and GEMM2
        gemm1_alpha_value: If provided, sets the swiglu alpha for all experts (float32)
        gemm1_beta_value: If provided, sets the linear offset (e.g., +1 in GPT OSS)
        gemm1_clamp_limit_value: If provided, sets the clamp limit for activation inputs
    """
    # Calculate padded sizes for trtllm_fp4_block_scale_moe alignment requirement
    padded_hidden_size = pad_to_multiple(hidden_size, ALIGNMENT)
    padded_intermediate_size = pad_to_multiple(intermediate_size, ALIGNMENT)


    print(f"\n{'='*70}")
    print(f"Testing MxFP4 x BF16 Fused MoE")
    print(f"  num_tokens={num_tokens}, hidden_size={hidden_size}")
    print(f"  intermediate_size={intermediate_size}, num_experts={num_experts}")
    print(f"  top_k={top_k}")
    print(f"  use_gemm_bias={use_gemm_bias}, alpha={gemm1_alpha_value}")
    print(f"  beta={gemm1_beta_value}, clamp_limit={gemm1_clamp_limit_value}")
    print(f"  [PADDING] hidden: {hidden_size} -> {padded_hidden_size}, "
            f"intermediate: {intermediate_size} -> {padded_intermediate_size}")
    print(f"{'='*70}")

    torch.cuda.synchronize()
    torch.random.manual_seed(0)

    moe_impl = FP4Moe()

    # Validation checks
    assert top_k <= num_experts, f"top_k ({top_k}) must be <= num_experts ({num_experts})"
    assert top_k <= 10, "top_k must be <= 10"

    # Create test data with ORIGINAL (unpadded) dimensions
    expert_logits = torch.randn((num_tokens, num_experts), device="cuda").to(
        torch.bfloat16
    )

    hidden_states = 2 * torch.randn(
        (num_tokens, hidden_size), device="cuda", dtype=torch.bfloat16
    )
    gemm1_weights = torch.randn(
        (num_experts, 2 * intermediate_size, hidden_size),
        device="cuda",
        dtype=torch.bfloat16,
    )
    gemm2_weights = torch.randn(
        (num_experts, hidden_size, intermediate_size),
        device="cuda",
        dtype=torch.bfloat16,
    )

    # Create bias tensors if enabled (with ORIGINAL dimensions)
    # IMPORTANT: FlashInfer expects bias tensors in float32 format.
    # Using bfloat16 will cause non-finite values in the output.
    if use_gemm_bias:
        gemm1_bias = torch.randn(
            (num_experts, 2 * intermediate_size),
            device="cuda",
            dtype=torch.float32,
        )
        gemm2_bias = torch.randn(
            (num_experts, hidden_size),
            device="cuda",
            dtype=torch.float32,
        )
    else:
        gemm1_bias = None
        gemm2_bias = None

    # Create alpha tensor if enabled
    # Shape: [num_experts], dtype: float32
    # Alpha scales the input to sigmoid in SwiGLU: x * sigmoid(alpha * x)
    # Standard SwiGLU uses alpha=1.0, GPT OSS uses alpha=1.702
    if gemm1_alpha_value is not None:
        gemm1_alpha = torch.full(
            (num_experts,),
            gemm1_alpha_value,
            device="cuda",
            dtype=torch.float32,
        )
    else:
        gemm1_alpha = None

    # Create beta tensor if enabled
    # Shape: [num_experts], dtype: float32
    # Beta is the offset added to the linear part: out_glu * (x_linear + beta)
    # GPT OSS uses beta=1.0
    if gemm1_beta_value is not None:
        gemm1_beta = torch.full(
            (num_experts,),
            gemm1_beta_value,
            device="cuda",
            dtype=torch.float32,
        )
    else:
        gemm1_beta = None

    # Create clamp_limit tensor if enabled
    # Shape: [num_experts], dtype: float32
    # Clamps the activation inputs: x_glu.clamp(max=limit), x_linear.clamp(-limit, limit)
    if gemm1_clamp_limit_value is not None:
        gemm1_clamp_limit = torch.full(
            (num_experts,),
            gemm1_clamp_limit_value,
            device="cuda",
            dtype=torch.float32,
        )
    else:
        gemm1_clamp_limit = None

    permute_info, scores = routing_reference_renormalize(
        expert_logits, top_k, num_experts, padding
    )

    # ===========================================================================
    # REFERENCE: Use original (unpadded) weights and activations
    # ===========================================================================

    # 1. Quantize original weights for reference
    print("Quantizing weights for reference...")
    (
        gemm1_weights_quant,
        gemm1_scales_quant,
        gemm2_weights_quant,
        gemm2_scales_quant,
    ) = moe_impl.quantize_weights(gemm1_weights, gemm2_weights)

    # 2. Convert hidden states to BF16
    hidden_states_bf16 = hidden_states.to(torch.bfloat16)

    # Compute reference output with original dimensions
    print("Computing reference output (unpadded)...")
    output_reference = run_moe_reference_fp4_bf16(
        num_tokens,
        num_experts,
        hidden_size,
        intermediate_size,
        top_k,
        padding,
        hidden_states_bf16,
        gemm1_weights_quant,
        gemm1_scales_quant,
        gemm2_weights_quant,
        gemm2_scales_quant,
        permute_info,
        gemm1_bias=gemm1_bias,
        gemm2_bias=gemm2_bias,
        gemm1_alpha=gemm1_alpha,
        gemm1_beta=gemm1_beta,
        gemm1_clamp_limit=gemm1_clamp_limit,
    )

    if output_reference is None:
        raise RuntimeError("Reference computation failed to produce output")

    # ===========================================================================
    # KERNEL: Use padded weights and activations
    # ===========================================================================

    # 3. Pad weights for kernel (if needed)
    print("Padding weights for kernel...")
    gemm1_weights_padded = pad_gate_up_weights(
        gemm1_weights, padded_hidden_size, padded_intermediate_size
    )
    gemm2_weights_padded = pad_down_weights(
        gemm2_weights, padded_hidden_size, padded_intermediate_size
    )

    # 4. Quantize padded weights for kernel
    print("Quantizing padded weights for kernel...")
    (
        gemm1_weights_quant_padded,
        _,
        gemm2_weights_quant_padded,
        _,
    ) = moe_impl.quantize_weights(gemm1_weights_padded, gemm2_weights_padded)

    # 5. Pad hidden states for kernel
    hidden_states_padded = pad_hidden_states(hidden_states, padded_hidden_size)

    # 6. Pad bias tensors for kernel (if provided)
    gemm1_bias_padded = None
    gemm2_bias_padded = None
    if gemm1_bias is not None:
        gemm1_bias_padded = pad_gate_up_bias(gemm1_bias, padded_intermediate_size)
    if gemm2_bias is not None:
        gemm2_bias_padded = pad_down_bias(gemm2_bias, padded_hidden_size)

    # 7. Prepare static weights for kernel with padded dimensions (includes bias shuffling)
    print("Preparing padded weights for kernel...")
    (
        gemm1_weights_shuffled,
        gemm1_scales_shuffled,
        gemm2_weights_shuffled,
        gemm2_scales_shuffled,
        gemm1_bias_shuffled,
        gemm2_bias_shuffled,
    ) = moe_impl.prepare_static_weights_for_kernel(
        gemm1_weights_quant_padded,
        gemm2_weights_quant_padded,
        gemm1_weights_padded,
        gemm2_weights_padded,
        padded_hidden_size,
        padded_intermediate_size,
        num_experts,
        gemm1_bias_padded=gemm1_bias_padded,
        gemm2_bias_padded=gemm2_bias_padded,
    )

    # Compute actual output using FlashInfer kernel with padded data
    print("Computing FlashInfer kernel output (padded)...")
    output_actual_padded = moe_impl.call_moe(
        gemm1_weights_shuffled,
        gemm1_scales_shuffled,
        gemm2_weights_shuffled,
        gemm2_scales_shuffled,
        hidden_states_padded,
        expert_logits=expert_logits,
        num_experts=num_experts,
        top_k=top_k,
        intermediate_size=padded_intermediate_size,
        gemm1_bias=gemm1_bias_shuffled,
        gemm2_bias=gemm2_bias_shuffled,
        gemm1_alpha=gemm1_alpha,
        gemm1_beta=gemm1_beta,
        gemm1_clamp_limit=gemm1_clamp_limit,
    )

    # 8. Strip padding from kernel output
    # Output shape is [num_tokens, padded_hidden_size], we need [num_tokens, hidden_size]
    output_actual = output_actual_padded[:, :hidden_size]

    # Compare outputs
    print("Checking accuracy...")
    tolerances = moe_impl.get_tolerances()
    check_accuracy(
        output_reference,
        output_actual,
        atol=tolerances["atol"],
        rtol=tolerances["rtol"],
        percent=tolerances["percent"],
    )

    print(f"✓ Test PASSED!")
    return True


# ====================================================================================
# Test with Dumped Weights from Model
# ====================================================================================


def deinterleave_gate_up_weights(weights: torch.Tensor) -> torch.Tensor:
    """
    De-interleave gate_up weights from GPT OSS format to FlashInfer format.
    
    GPT OSS stores gate_up as interleaved: [gate[0], linear[0], gate[1], linear[1], ...]
    FlashInfer expects non-interleaved: [linear[0], linear[1], ..., gate[0], gate[1], ...]
    """
    num_experts, fused_size, hidden_size = weights.shape
    
    # Extract interleaved parts: gate at even indices, linear at odd indices
    gate_part = weights[:, 0::2, :]   # [num_experts, intermediate_size, hidden_size]
    linear_part = weights[:, 1::2, :] # [num_experts, intermediate_size, hidden_size]
    
    # Concatenate: linear first, gate second (FlashInfer format)
    return torch.cat([linear_part, gate_part], dim=1)


def deinterleave_gate_up_bias(bias: torch.Tensor) -> torch.Tensor:
    """
    De-interleave gate_up bias from GPT OSS format to FlashInfer format.
    """
    num_experts, fused_size = bias.shape
    
    # Extract interleaved parts
    gate_part = bias[:, 0::2]   # [num_experts, intermediate_size]
    linear_part = bias[:, 1::2] # [num_experts, intermediate_size]
    
    # Concatenate: linear first, gate second (FlashInfer format)
    return torch.cat([linear_part, gate_part], dim=1)


def run_moe_test_with_dumped_weights(
    dump_dir: str = "/tmp/moe_debug",
    padding: int = 8,
):
    """
    Test MxFP4 x BF16 fused MoE using weights dumped from the model.
    
    This loads weights from the specified directory and runs the same test
    as run_moe_test but with actual model weights instead of random ones.
    """
    import os
    
    print(f"\n{'='*70}")
    print(f"Testing with dumped weights from: {dump_dir}")
    print(f"{'='*70}")
    
    # Load dumped data
    hidden_states = torch.load(f"{dump_dir}/hidden_states.pt", weights_only=True)
    router_logits = torch.load(f"{dump_dir}/router_logits.pt", weights_only=True)
    gate_up_proj = torch.load(f"{dump_dir}/gate_up_proj.pt", weights_only=True)
    gate_up_proj_bias = torch.load(f"{dump_dir}/gate_up_proj_bias.pt", weights_only=True)
    down_proj = torch.load(f"{dump_dir}/down_proj.pt", weights_only=True)
    down_proj_bias = torch.load(f"{dump_dir}/down_proj_bias.pt", weights_only=True)
    config_info = torch.load(f"{dump_dir}/config_info.pt", weights_only=True)

    num_experts = 32
    hidden_size = 2880
    num_tokens = 146
    intermediate_size = 2880

    # hidden_states = 2 * torch.randn(
    #     (num_tokens, hidden_size), device="cuda", dtype=torch.bfloat16
    # )
    # router_logits = torch.randn((num_tokens, num_experts), device="cuda").to(
    #     torch.bfloat16
    # )
    # gate_up_proj = torch.randn(
    #     (num_experts, 2 * intermediate_size, hidden_size),
    #     device="cuda",
    #     dtype=torch.bfloat16,
    # )
    # down_proj = torch.randn(
    #     (num_experts, hidden_size, intermediate_size),
    #     device="cuda",
    #     dtype=torch.bfloat16,
    # )
    # gate_up_proj_bias = torch.randn(
    #         (num_experts, 2 * intermediate_size),
    #         device="cuda",
    #         dtype=torch.float32,
    #     )
    # down_proj_bias = torch.randn(
    #     (num_experts, hidden_size),
    #     device="cuda",
    #     dtype=torch.float32,
    # )

    # assert num_experts == config_info["num_experts"]
    # assert hidden_size == config_info["hidden_size"]
    # assert intermediate_size == config_info["intermediate_size"]
    # assert num_tokens == hidden_states.shape[0]
    
    # num_experts = config_info["num_experts"]
    # hidden_size = config_info["hidden_size"]
    # intermediate_size = config_info["intermediate_size"]
    # num_tokens = hidden_states.shape[0]

    top_k = config_info["experts_per_token"]
    swiglu_limit = config_info["swiglu_limit"]
    
    print(f"  num_tokens={num_tokens}, hidden_size={hidden_size}")
    print(f"  intermediate_size={intermediate_size}, num_experts={num_experts}")
    print(f"  top_k={top_k}, swiglu_limit={swiglu_limit}")
    
    # Calculate padded sizes
    padded_hidden_size = pad_to_multiple(hidden_size, ALIGNMENT)
    padded_intermediate_size = pad_to_multiple(intermediate_size, ALIGNMENT)
    print(f"  [PADDING] hidden: {hidden_size} -> {padded_hidden_size}, "
          f"intermediate: {intermediate_size} -> {padded_intermediate_size}")
    
    # De-interleave weights from GPT OSS format to FlashInfer format
    print("De-interleaving gate_up weights...")
    gemm1_weights = deinterleave_gate_up_weights(gate_up_proj)
    gemm1_bias = deinterleave_gate_up_bias(gate_up_proj_bias)
    gemm2_weights = down_proj
    gemm2_bias = down_proj_bias
    
    # Use router_logits as expert_logits
    expert_logits = router_logits.to(torch.bfloat16)
    
    # Prepare activation parameters (GPT OSS style)
    gemm1_alpha = torch.full((num_experts,), 1.702, device="cuda", dtype=torch.float32)
    gemm1_beta = torch.full((num_experts,), 1.0, device="cuda", dtype=torch.float32)
    gemm1_clamp_limit = torch.full((num_experts,), swiglu_limit, device="cuda", dtype=torch.float32)
    
    # Convert bias to float32 as expected by FlashInfer
    gemm1_bias = gemm1_bias.float()
    gemm2_bias = gemm2_bias.float()
    
    moe_impl = FP4Moe()
    
    # Get permute info
    permute_info, scores = routing_reference_renormalize(
        expert_logits, top_k, num_experts, padding
    )
    
    # ===========================================================================
    # PAD WEIGHTS FIRST (before quantization)
    # ===========================================================================
    print("Padding weights...")
    gemm1_weights_padded = pad_gate_up_weights(
        gemm1_weights.to(torch.bfloat16), padded_hidden_size, padded_intermediate_size
    )
    gemm2_weights_padded = pad_down_weights(
        gemm2_weights.to(torch.bfloat16), padded_hidden_size, padded_intermediate_size
    )
    
    # Pad hidden states
    hidden_states_padded = pad_hidden_states(hidden_states.to(torch.bfloat16), padded_hidden_size)
    
    # Pad bias tensors
    gemm1_bias_padded = pad_gate_up_bias(gemm1_bias, padded_intermediate_size)
    gemm2_bias_padded = pad_down_bias(gemm2_bias, padded_hidden_size)
    
    # ===========================================================================
    # QUANTIZE PADDED WEIGHTS
    # ===========================================================================
    print("Quantizing padded weights...")
    (
        gemm1_weights_quant_padded,
        gemm1_scales_quant_padded,
        gemm2_weights_quant_padded,
        gemm2_scales_quant_padded,
    ) = moe_impl.quantize_weights(gemm1_weights_padded, gemm2_weights_padded)
    
    # ===========================================================================
    # REFERENCE: Use padded weights and dimensions
    # ===========================================================================
    print("Computing reference output (padded)...")
    output_reference_padded = run_moe_reference_fp4_bf16(
        num_tokens,
        num_experts,
        padded_hidden_size,
        padded_intermediate_size,
        top_k,
        padding,
        hidden_states_padded,
        gemm1_weights_quant_padded,
        gemm1_scales_quant_padded,
        gemm2_weights_quant_padded,
        gemm2_scales_quant_padded,
        permute_info,
        gemm1_bias=gemm1_bias_padded,
        gemm2_bias=gemm2_bias_padded,
        # gemm1_bias=None,
        # gemm2_bias=None,
        gemm1_alpha=gemm1_alpha,
        gemm1_beta=gemm1_beta,
        gemm1_clamp_limit=gemm1_clamp_limit,
        # gemm1_alpha=None,
        # gemm1_beta=None,
        # gemm1_clamp_limit=None,
    )
    # Strip padding from reference output
    output_reference = output_reference_padded[:, :hidden_size]
    
    # ===========================================================================
    # KERNEL: Prepare shuffled weights (includes bias shuffling)
    # ===========================================================================
    print("Preparing shuffled weights for kernel...")
    (
        gemm1_weights_shuffled,
        gemm1_scales_shuffled,
        gemm2_weights_shuffled,
        gemm2_scales_shuffled,
        gemm1_bias_shuffled,
        gemm2_bias_shuffled,
    ) = moe_impl.prepare_static_weights_for_kernel(
        gemm1_weights_quant_padded,
        gemm2_weights_quant_padded,
        gemm1_weights_padded,
        gemm2_weights_padded,
        padded_hidden_size,
        padded_intermediate_size,
        num_experts,
        gemm1_bias_padded=gemm1_bias_padded,
        gemm2_bias_padded=gemm2_bias_padded,
    )
    
    print("Computing FlashInfer kernel output (padded)...")
    output_actual_padded = moe_impl.call_moe(
        gemm1_weights_shuffled,
        gemm1_scales_shuffled,
        gemm2_weights_shuffled,
        gemm2_scales_shuffled,
        hidden_states_padded,
        expert_logits=expert_logits,
        num_experts=num_experts,
        top_k=top_k,
        intermediate_size=padded_intermediate_size,
        gemm1_bias=gemm1_bias_shuffled,
        gemm2_bias=gemm2_bias_shuffled,
        # gemm1_bias=None,
        # gemm2_bias=None,
        gemm1_alpha=gemm1_alpha,
        gemm1_beta=gemm1_beta,
        gemm1_clamp_limit=gemm1_clamp_limit,
        # gemm1_alpha=None,
        # gemm1_beta=None,
        # gemm1_clamp_limit=None,
    )
    
    # Strip padding from output
    output_actual = output_actual_padded[:, :hidden_size]
    
    # Compare outputs
    print("Checking accuracy...")
    tolerances = moe_impl.get_tolerances()
    
    # Detailed comparison
    diff = (output_reference - output_actual).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    close = torch.isclose(output_reference, output_actual, atol=tolerances["atol"], rtol=tolerances["rtol"])
    match_ratio = close.float().mean().item()
    
    print(f"  max diff: {max_diff:.6f}")
    print(f"  mean diff: {mean_diff:.6f}")
    print(f"  match ratio (atol={tolerances['atol']}, rtol={tolerances['rtol']}): {match_ratio:.4f}")
    print(f"  required: {tolerances['percent']:.4f}")
    
    print(f"Sample outputs (first 5 elements of first token):")
    print(f"  reference:  {output_reference[0, :5].tolist()}")
    print(f"  FlashInfer: {output_actual[0, :5].tolist()}")
    
    if match_ratio >= tolerances["percent"]:
        print(f"✓ Test PASSED!")
        return True
    else:
        print(f"✗ Test FAILED!")
        return False


# ====================================================================================
# Main Entry Point
# ====================================================================================


if __name__ == "__main__":
    import sys
    
    # Check if we should test with dumped weights
    if len(sys.argv) > 1 and sys.argv[1] == "--dumped":
        dump_dir = sys.argv[2] if len(sys.argv) > 2 else "/tmp/moe_debug"
        print(f"Testing with dumped weights from: {dump_dir}")
        try:
            success = run_moe_test_with_dumped_weights(dump_dir)
            if not success:
                exit(1)
        except Exception as e:
            print(f"\n✗ Test FAILED: {e}")
            import traceback
            traceback.print_exc()
            exit(1)
        exit(0)
    
    print("FlashInfer Fused MoE Sanity Check")
    print("Testing MxFP4 weights with BF16 activation")
    print("=" * 70)

    try:
        # Test 1: Basic Renormalize routing (Qwen3-style) - no bias
        run_moe_test(
            num_tokens=128,
            hidden_size=1024,
            intermediate_size=1024,
            num_experts=128,
            top_k=8,
            padding=8,
            use_gemm_bias=False,
        )

        # Test 2: Smaller configuration - no bias
        run_moe_test(
            num_tokens=8,
            hidden_size=1024,
            intermediate_size=768,
            num_experts=128,
            top_k=8,
            padding=8,
            use_gemm_bias=False,
        )

        # Test 3: With GEMM bias - larger configuration
        run_moe_test(
            num_tokens=128,
            hidden_size=1024,
            intermediate_size=1024,
            num_experts=128,
            top_k=8,
            padding=8,
            use_gemm_bias=True,
        )

        # Test 4: With GEMM bias - smaller configuration
        run_moe_test(
            num_tokens=8,
            hidden_size=1024,
            intermediate_size=768,
            num_experts=128,
            top_k=8,
            padding=8,
            use_gemm_bias=True,
        )

        # Test 5: With alpha=1.702 (GPT OSS style) - no bias
        run_moe_test(
            num_tokens=128,
            hidden_size=1024,
            intermediate_size=1024,
            num_experts=128,
            top_k=8,
            padding=8,
            use_gemm_bias=False,
            gemm1_alpha_value=1.702,
        )

        # Test 6: With alpha=1.702 and GEMM bias (GPT OSS style)
        run_moe_test(
            num_tokens=128,
            hidden_size=1024,
            intermediate_size=1024,
            num_experts=128,
            top_k=8,
            padding=8,
            use_gemm_bias=True,
            gemm1_alpha_value=1.702,
        )

        # Test 7: With beta=1.0 (GPT OSS style offset)
        run_moe_test(
            num_tokens=128,
            hidden_size=1024,
            intermediate_size=1024,
            num_experts=128,
            top_k=8,
            padding=8,
            use_gemm_bias=False,
            gemm1_alpha_value=1.702,
            gemm1_beta_value=1.0,
        )

        # Test 8: With clamp_limit=7.0 (GPT OSS style clamping)
        run_moe_test(
            num_tokens=128,
            hidden_size=1024,
            intermediate_size=1024,
            num_experts=128,
            top_k=8,
            padding=8,
            use_gemm_bias=False,
            gemm1_alpha_value=1.702,
            gemm1_clamp_limit_value=7.0,
        )

        # Test 9: Full GPT OSS configuration (bias + alpha + beta + clamp_limit)
        run_moe_test(
            num_tokens=128,
            hidden_size=1024,
            intermediate_size=1024,
            num_experts=128,
            top_k=8,
            padding=8,
            use_gemm_bias=True,
            gemm1_alpha_value=1.702,
            gemm1_beta_value=1.0,
            gemm1_clamp_limit_value=7.0,
        )

        # Test 10: Full GPT OSS configuration that requires padding (bias + alpha + beta + clamp_limit)
        run_moe_test(
            num_tokens=128,
            hidden_size=2944,
            intermediate_size=2944,
            num_experts=128,
            top_k=8,
            padding=8,
            use_gemm_bias=True,
            gemm1_alpha_value=1.702,
            gemm1_beta_value=1.0,
            gemm1_clamp_limit_value=7.0,
        )

        print("\n" + "=" * 70)
        print("All tests PASSED!")
        print("=" * 70)

    except Exception as e:
        print(f"\n✗ Test FAILED: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
