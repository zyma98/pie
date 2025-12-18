"""
Sanity check script for FlashInfer fused MoE with MXFP4 weights and BF16 activation.

This script tests our understanding of the FlashInfer `trtllm_fp4_block_scale_moe` function
by comparing its output against a reference dequantized implementation.

Based on the official FlashInfer tests:
https://github.com/flashinfer-ai/flashinfer/blob/0e68a2febc58df99429f652769c5c485ca67fc39/tests/moe/test_trtllm_gen_fused_moe.py
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import IntEnum
from typing import Dict

import pytest
import torch
from torch.nn import functional as F

from flashinfer import (
    GatedActType,
    RoutingMethodType,
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


class QuantMode(IntEnum):
    """Supported quantization modes for MoE testing."""

    FP4_NVFP4_NVFP4 = 1
    FP4_MXFP4_MXFP8 = 2
    FP4_MXFP4_Bf16 = 3
    FP8_BLOCK_SCALE = 4
    FP8_PER_TENSOR = 5
    BF16 = 6


# ====================================================================================
# FP4 Quantization Functions
# ====================================================================================


def calculate_fp4_global_scale_factor(tensor, use_ue8m0=False):
    """
    Calculate FP4 global scale factor for a tensor.

    NOTE: In production, global scale factors are typically obtained offline during:
    - Post-Training Quantization (PTQ) calibration process
    - Quantization-Aware Training (QAT) process

    This function is used here for testing/reference purposes.
    Formula: (448 * 6) represents max representable value in FP4 format.
    """
    if use_ue8m0:
        return torch.tensor(1.0, dtype=torch.float32)
    else:
        return (448 * 6) / tensor.float().abs().nan_to_num().max()


def quant_fp4(a, a_global_sf, use_ue8m0=False, is_sf_swizzled_layout=True):
    """
    Quantize FP4 with pre-computed global scale factor.

    Pure function - same inputs always produce same outputs.
    """
    sf_vec_size = 32 if use_ue8m0 else 16

    a_fp4, a_sf = fp4_quantize(
        a.cuda(), a_global_sf.cuda(), sf_vec_size, use_ue8m0, is_sf_swizzled_layout
    )

    return a_fp4, a_sf, a_global_sf


def quant_fp4_batches(a, num_experts, use_ue8m0=False, is_sf_swizzled_layout=True):
    """FP4 batch quantization function with centralized global scale factor calculation."""
    quant_a = []
    sfs = []
    global_sfs = []
    for i in range(num_experts):
        # Use centralized global scale factor calculation
        a_global_sf = calculate_fp4_global_scale_factor(a[i], use_ue8m0)
        a_fp4, a_sf, _ = quant_fp4(a[i], a_global_sf, use_ue8m0, is_sf_swizzled_layout)
        quant_a.append(a_fp4)
        sfs.append(a_sf)
        global_sfs.append(a_global_sf)

    result_quant_a = torch.stack(quant_a)
    result_sfs = torch.stack(sfs)
    result_global_sfs = torch.stack(global_sfs)

    return result_quant_a, result_sfs, result_global_sfs


def e2m1_and_ufp8_scale_batches(
    mat_fp4: torch.Tensor,
    scale_tensor: torch.Tensor,
    global_scale_tensor: torch.Tensor,
    sf_vec_size: int,
    ufp8_type: int = 1,
):
    """Batch FP4 dequantization helper."""
    num_batches = mat_fp4.size(0)
    scale_tensor = scale_tensor.view(num_batches, -1)

    tensors = [
        e2m1_and_ufp8sf_scale_to_float(
            mat_fp4[b, :, :].cpu(),
            scale_tensor[b, :].cpu().reshape(-1),
            global_scale_tensor[b].cpu(),
            sf_vec_size,
            ufp8_type,
            True,  # is_sf_swizzled_layout
        )
        for b in range(num_batches)
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


def noaux_tc_ref(logits, bias, n_group, topk_group, top_k, routed_scaling_factor):
    """DeepSeek-style no-aux routing reference implementation."""
    scores = F.sigmoid(logits)
    scores_with_bias = scores + bias
    if n_group > 1:
        scores_shape = list(scores_with_bias.shape)
        group_scores = torch.sum(
            torch.topk(
                scores_with_bias.view(
                    scores_shape[:-1] + [n_group, scores_shape[-1] // n_group]
                ),
                k=2,
                dim=-1,
                largest=True,
                sorted=True,
            )[0],
            dim=-1,
        )
        _, group_idx = torch.topk(
            group_scores, k=topk_group, dim=-1, largest=True, sorted=True
        )
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(-1, group_idx, 1)
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(scores_shape[:-1] + [n_group, scores_shape[-1] // n_group])
            .reshape(scores_shape)
        )
        scores_with_bias = scores_with_bias * score_mask

    _, topk_idx = torch.topk(
        scores_with_bias, k=top_k, dim=-1, largest=True, sorted=True
    )
    new_mask = torch.zeros_like(scores)
    new_mask.scatter_(-1, topk_idx, 1)
    scores = scores * new_mask
    score_sum = torch.sum(scores, dim=-1, keepdim=True) + 1e-20
    scores = scores / score_sum * routed_scaling_factor
    return scores


def routing_reference_no_aux(
    expert_logits,
    routing_bias,
    top_k,
    n_groups,
    top_k_groups,
    routed_scaling,
    padding,
    use_routing_scales_on_input=False,
):
    """Tiered TopK routing used by DeepSeek."""
    routing_logits = expert_logits.to(dtype=torch.float, device="cuda")
    if use_routing_scales_on_input:
        # if using routing scales on input, topK == 1 and the score is a plain sigmoid
        scores = F.sigmoid(routing_logits)
    else:
        scores = noaux_tc_ref(
            routing_logits, routing_bias, n_groups, top_k_groups, top_k, routed_scaling
        )
    permute_info = routing_reference(scores, top_k, padding)
    return permute_info, scores


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
# MoE Arguments Containers
# ====================================================================================


class moe_args:
    """Arguments container for MoE operations."""

    def __init__(
        self,
        num_tokens,
        num_experts,
        hidden_size,
        intermediate_size,
        top_k,
        padding,
        hidden_states,
        hidden_states_scale,
        hidden_states_scale_global,
        expert_logits,
        gemm1_weights,
        gemm1_scales,
        gemm1_scales_global,
        gemm2_weights,
        gemm2_scales,
        gemm2_scales_global,
        permute_info,
        use_routing_scales_on_input,
        gated_act_type,
        gemm1_bias=None,
        gemm2_bias=None,
        gemm1_alpha=None,
        gemm1_beta=None,
        gemm1_clamp_limit=None,
    ):
        self.num_tokens = num_tokens
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.top_k = top_k
        self.padding = padding
        self.hidden_states = hidden_states
        self.hidden_states_scale = hidden_states_scale
        self.hidden_states_scale_global = hidden_states_scale_global
        self.expert_logits = expert_logits
        self.gemm1_weights = gemm1_weights
        self.gemm1_scales = gemm1_scales
        self.gemm1_scales_global = gemm1_scales_global
        self.gemm2_weights = gemm2_weights
        self.gemm2_scales = gemm2_scales
        self.gemm2_scales_global = gemm2_scales_global
        self.permute_info = permute_info
        self.use_routing_scales_on_input = use_routing_scales_on_input
        self.gated_act_type = gated_act_type
        self.gemm1_bias = gemm1_bias
        self.gemm2_bias = gemm2_bias
        self.gemm1_alpha = gemm1_alpha
        self.gemm1_beta = gemm1_beta
        self.gemm1_clamp_limit = gemm1_clamp_limit


class moe_args_dequant:
    """Arguments container for dequantized MoE operations."""

    def __init__(
        self,
        num_tokens,
        num_experts,
        hidden_size,
        intermediate_size,
        top_k,
        padding,
        hidden_states,
        expert_logits,
        gemm1_weights,
        gemm2_weights,
        permute_info,
        use_routing_scales_on_input,
        gated_act_type,
        hidden_states_scale=None,
        gemm1_bias=None,
        gemm2_bias=None,
        gemm1_alpha=None,
        gemm1_beta=None,
        gemm1_clamp_limit=None,
    ):
        self.num_tokens = num_tokens
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.top_k = top_k
        self.padding = padding
        self.hidden_states = hidden_states
        self.expert_logits = expert_logits
        self.gemm1_weights = gemm1_weights
        self.gemm2_weights = gemm2_weights
        self.permute_info = permute_info
        self.use_routing_scales_on_input = use_routing_scales_on_input
        self.gated_act_type = gated_act_type
        self.hidden_states_scale = hidden_states_scale
        self.gemm1_bias = gemm1_bias
        self.gemm2_bias = gemm2_bias
        self.gemm1_alpha = gemm1_alpha
        self.gemm1_beta = gemm1_beta
        self.gemm1_clamp_limit = gemm1_clamp_limit


# ====================================================================================
# Common MoE Reference Implementation
# ====================================================================================


def run_moe_dequant(args, quant_mode: QuantMode):
    """Common dequantized MoE reference implementation."""
    # Permute
    total_num_padded_tokens = args.permute_info["permutedBufferSize"]
    expanded_idx_to_permuted_idx = args.permute_info[
        "expandedTokenIdxToPermutedIdx"
    ].cpu()
    num_tokens_per_expert = args.permute_info["numTokensPerExpert"].cpu()
    permute_output = torch.full(
        (total_num_padded_tokens, args.hidden_size), float("nan"), device="cuda"
    ).to(torch.float)
    for i in range(args.num_tokens):
        for j in range(args.top_k):
            permuted_idx = expanded_idx_to_permuted_idx[i * args.top_k + j]
            permute_output[permuted_idx] = args.hidden_states[i]

    # Gemm1
    gemm1_output = torch.full(
        (total_num_padded_tokens, 2 * args.intermediate_size),
        float("nan"),
        device="cuda",
    ).to(torch.float)
    i = 0
    for expert_idx in range(args.num_experts):
        my_num_tokens = num_tokens_per_expert[expert_idx]
        if my_num_tokens == 0:
            continue
        my_a = permute_output[i : i + my_num_tokens]
        my_b = args.gemm1_weights[expert_idx]
        my_c = my_a @ my_b.t()
        # Add gemm1 bias if provided
        if args.gemm1_bias is not None:
            my_c = my_c + args.gemm1_bias[expert_idx]
        gemm1_output[i : i + my_num_tokens] = my_c
        i += my_num_tokens
        i = (i + args.padding - 1) // args.padding * args.padding

    if args.use_routing_scales_on_input:
        assert args.top_k == 1
        # For each token and its top_k experts
        for token_idx in range(args.num_tokens):
            for k in range(args.top_k):
                # Get the permuted index for this token's k-th expert
                expanded_idx = token_idx * args.top_k + k
                permuted_idx = expanded_idx_to_permuted_idx[expanded_idx]
                expert_weight = args.permute_info["topKLogits"].to(torch.float)
                # Get the expert weight for this token and expert
                weight = expert_weight[token_idx, k]
                # Scale the corresponding row in gemm1_output
                gemm1_output[permuted_idx] *= weight

    # Activation
    activation_output = torch.full(
        (total_num_padded_tokens, args.intermediate_size), float("nan"), device="cuda"
    ).to(torch.float)

    gated_act_type = args.gated_act_type
    gated_act_type_to_func = {
        0: F.silu,
        1: F.gelu,
    }
    gated_act_func = gated_act_type_to_func[gated_act_type]

    i = 0
    for expert_idx in range(args.num_experts):
        my_num_tokens = num_tokens_per_expert[expert_idx]
        if my_num_tokens == 0:
            continue
        my_a = gemm1_output[i : i + my_num_tokens]
        my_x1 = my_a[:, : args.intermediate_size]  # linear part
        my_x2 = my_a[:, args.intermediate_size :]  # gated part

        # Apply clamping if clamp_limit is provided (GPT OSS style)
        # x_glu (gated) clamped to max=limit
        # x_linear clamped to [-limit, limit]
        if args.gemm1_clamp_limit is not None:
            clamp_limit = args.gemm1_clamp_limit[expert_idx]
            my_x2 = my_x2.clamp(max=clamp_limit)
            my_x1 = my_x1.clamp(min=-clamp_limit, max=clamp_limit)

        # Apply activation with optional alpha scaling
        # Standard SwiGLU: silu(x2) * x1 = (x2 * sigmoid(x2)) * x1
        # With alpha: (x2 * sigmoid(alpha * x2)) * x1
        if args.gemm1_alpha is not None:
            alpha = args.gemm1_alpha[expert_idx]
            gated_output = my_x2 * torch.sigmoid(alpha * my_x2)
        else:
            gated_output = gated_act_func(my_x2)

        # Apply beta offset to linear part (GPT OSS style: out_glu * (x_linear + beta))
        if args.gemm1_beta is not None:
            beta = args.gemm1_beta[expert_idx]
            linear_output = my_x1 + beta
        else:
            linear_output = my_x1

        activation_output[i : i + my_num_tokens] = gated_output * linear_output
        i += my_num_tokens
        i = (i + args.padding - 1) // args.padding * args.padding

    # For MxFP4xBf16, just convert activation to bf16 and back
    if quant_mode == QuantMode.FP4_MXFP4_Bf16:
        activation_output = activation_output.to(torch.bfloat16).to(torch.float)
        args.c_global_sf = 1.0

    # Gemm2
    gemm2_output = torch.full(
        (total_num_padded_tokens, args.hidden_size), float("nan"), device="cuda"
    ).to(torch.float)
    i = 0
    for expert_idx in range(args.num_experts):
        my_num_tokens = num_tokens_per_expert[expert_idx]
        if my_num_tokens == 0:
            continue
        my_a = activation_output[i : i + my_num_tokens]
        my_b = args.gemm2_weights[expert_idx]
        my_c = my_a @ my_b.t()
        # Add gemm2 bias if provided
        if args.gemm2_bias is not None:
            my_c = my_c + args.gemm2_bias[expert_idx]
        gemm2_output[i : i + my_num_tokens] = my_c
        i += my_num_tokens
        i = (i + args.padding - 1) // args.padding * args.padding

    # Finalize
    expert_weight = args.permute_info["topKLogits"].to(torch.float)
    finalize_output = torch.full(
        (args.num_tokens, args.hidden_size), float("nan"), device="cuda"
    ).to(torch.float)
    for i in range(args.num_tokens):
        acc = torch.zeros(args.hidden_size, dtype=torch.float, device="cuda")
        for top_k_idx in range(args.top_k):
            expanded_idx = i * args.top_k + top_k_idx
            permuted_idx = expanded_idx_to_permuted_idx[expanded_idx]
            original_vector = gemm2_output[permuted_idx]
            weight = (
                expert_weight[i, top_k_idx]
                if not args.use_routing_scales_on_input
                else 1.0
            )
            acc += original_vector * weight
        finalize_output[i] = acc
    return finalize_output


def run_moe_reference_fp4_bf16(args, quant_mode: QuantMode):
    """FP4 with BF16 activation reference implementation."""
    sf_vec_size = 32  # MXFP4 uses 32
    ufp8_type_weights = 0  # MXFP4 uses ue8m0

    # Hidden states are already BF16, just convert to float for reference
    hidden_states_dequant = args.hidden_states.to(torch.bfloat16).to(torch.float)

    gemm1_weights_dequant = e2m1_and_ufp8_scale_batches(
        args.gemm1_weights,
        args.gemm1_scales,
        1 / args.gemm1_scales_global,
        sf_vec_size,
        ufp8_type_weights,
    ).cuda()

    gemm2_weights_dequant = e2m1_and_ufp8_scale_batches(
        args.gemm2_weights,
        args.gemm2_scales,
        1 / args.gemm2_scales_global,
        sf_vec_size,
        ufp8_type_weights,
    ).cuda()

    args_dequant = moe_args_dequant(
        args.num_tokens,
        args.num_experts,
        args.hidden_size,
        args.intermediate_size,
        args.top_k,
        args.padding,
        hidden_states_dequant,
        args.expert_logits,
        gemm1_weights_dequant,
        gemm2_weights_dequant,
        args.permute_info,
        args.use_routing_scales_on_input,
        args.gated_act_type,
        gemm1_bias=args.gemm1_bias,
        gemm2_bias=args.gemm2_bias,
        gemm1_alpha=args.gemm1_alpha,
        gemm1_beta=args.gemm1_beta,
        gemm1_clamp_limit=args.gemm1_clamp_limit,
    )

    return run_moe_dequant(args_dequant, quant_mode), args_dequant


# ====================================================================================
# FP4Moe Implementation for MxFP4 x Bf16
# ====================================================================================


class FP4Moe:
    """
    FP4 MxFP4 MoE implementation with block scaling and BF16 activation.
    """

    def __init__(self, quant_mode: QuantMode = QuantMode.FP4_MXFP4_Bf16):
        self.name = "FP4Moe"
        self.quant_mode = quant_mode
        self.is_mxfp4 = True
        self.sf_vec_size = 32  # MXFP4 uses 32-element blocks
        self._cache_permute_indices: Dict[tuple, torch.Tensor] = {}

    def quantize_weights(self, gemm1_weights, gemm2_weights, hidden_states_sample):
        """Quantize weights to FP4 format and compute global scale factors."""
        num_experts = gemm1_weights.shape[0]
        # BF16 hidden states, no global scale needed
        hidden_states_scale_global = 1.0

        # Quantize the weights for FC1
        gemm1_weights_fp4_bytes, gemm1_scales_fp4_bytes, gemm1_scales_global = (
            quant_fp4_batches(gemm1_weights, num_experts, self.is_mxfp4, True)
        )

        # Quantize the weights for FC2
        gemm2_weights_fp4_bytes, gemm2_scales_fp4_bytes, gemm2_scales_global = (
            quant_fp4_batches(gemm2_weights, num_experts, self.is_mxfp4, True)
        )

        return {
            "hidden_states_scale_global": hidden_states_scale_global,
            "gemm1_weights": gemm1_weights_fp4_bytes,
            "gemm1_scales": gemm1_scales_fp4_bytes,
            "gemm1_scales_global": gemm1_scales_global,
            "gemm2_weights": gemm2_weights_fp4_bytes,
            "gemm2_scales": gemm2_scales_fp4_bytes,
            "gemm2_scales_global": gemm2_scales_global,
        }

    def quantize_inputs(self, hidden_states, hidden_states_scale_global):
        """BF16 hidden states - no quantization needed."""
        return {
            "hidden_states": hidden_states.to(torch.bfloat16),
            "hidden_states_scale": None,
        }

    def prepare_static_weights_for_kernel(
        self,
        args_dequant,
        args,
        gemm1_weights_orig,
        gemm2_weights_orig,
        hidden_size,
        intermediate_size,
        num_experts,
    ):
        """Prepare quantized weights for kernel (done offline with weights)."""
        epilogue_tile_m = 128  # FIXME: this depends on the kernel internals

        # Quantize weights with linear layout for kernels
        _, gemm1_scales_linear_fp4_bytes, _ = quant_fp4_batches(
            gemm1_weights_orig, num_experts, self.is_mxfp4, False
        )
        _, gemm2_scales_linear_fp4_bytes, _ = quant_fp4_batches(
            gemm2_weights_orig, num_experts, self.is_mxfp4, False
        )

        # Convert quantized weights to proper formats
        gemm1_weights_fp4 = args.gemm1_weights.view(torch.float8_e4m3fn).reshape(
            num_experts, 2 * intermediate_size, hidden_size // 2
        )  # packed fp4
        gemm1_scales_linear_fp4 = gemm1_scales_linear_fp4_bytes.view(
            torch.float8_e4m3fn
        ).reshape(
            num_experts, 2 * intermediate_size, hidden_size // self.sf_vec_size
        )  # fp8 scaling factors

        gemm2_weights_fp4 = args.gemm2_weights.view(torch.float8_e4m3fn).reshape(
            num_experts, hidden_size, intermediate_size // 2
        )  # packed fp4
        gemm2_scales_linear_fp4 = gemm2_scales_linear_fp4_bytes.view(
            torch.float8_e4m3fn
        ).reshape(
            num_experts, hidden_size, intermediate_size // self.sf_vec_size
        )  # fp8 scaling factors

        # Using cached permute index calculation can speed up weights preprocessing
        gemm1_weights_fp4_shuffled = []
        gemm1_scales_fp4_shuffled = []
        gemm2_weights_fp4_shuffled = []
        gemm2_scales_fp4_shuffled = []
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
                num_experts, 2 * intermediate_size, hidden_size // self.sf_vec_size
            )
        )

        gemm2_weights_fp4_shuffled = torch.stack(gemm2_weights_fp4_shuffled)
        gemm2_scales_fp4_shuffled = (
            torch.stack(gemm2_scales_fp4_shuffled)
            .view(torch.float8_e4m3fn)
            .reshape(num_experts, hidden_size, intermediate_size // self.sf_vec_size)
        )

        # Calculate scaling factors that depend on weights
        scale_c_fc1 = (
            args_dequant.c_global_sf
            * (1.0 / args.gemm1_scales_global)
            * (1.0 / args.hidden_states_scale_global)
        )
        scale_gate_fc1 = (1.0 / args.gemm1_scales_global) * (
            1.0 / args.hidden_states_scale_global
        )
        scale_c_fc2 = (1.0 / args_dequant.c_global_sf) * (
            1.0 / args.gemm2_scales_global
        )

        return {
            "gemm1_weights_fp4_shuffled": gemm1_weights_fp4_shuffled,
            "gemm1_scales_fp4_shuffled": gemm1_scales_fp4_shuffled,
            "gemm2_weights_fp4_shuffled": gemm2_weights_fp4_shuffled,
            "gemm2_scales_fp4_shuffled": gemm2_scales_fp4_shuffled,
            "scale_c_fc1": scale_c_fc1,
            "scale_gate_fc1": scale_gate_fc1,
            "scale_c_fc2": scale_c_fc2,
        }

    def call_moe(
        self,
        static_data,
        hidden_states_orig,
        hidden_states_scale_global,
        **kwargs,
    ):
        """Call the FlashInfer fused MoE kernel."""
        expert_logits = kwargs["expert_logits"]
        routing_bias = kwargs["routing_bias"]
        num_experts = kwargs["num_experts"]
        top_k = kwargs["top_k"]
        n_groups = kwargs["n_groups"]
        top_k_groups = kwargs["top_k_groups"]
        intermediate_size = kwargs["intermediate_size"]
        routed_scaling = kwargs["routed_scaling"]
        gated_act_type = kwargs["gated_act_type"]
        routing_method_type = kwargs["routing_method_type"]
        enable_autotune = kwargs.get("enable_autotune", True)
        gemm1_bias = kwargs.get("gemm1_bias", None)
        gemm2_bias = kwargs.get("gemm2_bias", None)
        gemm1_alpha = kwargs.get("gemm1_alpha", None)
        gemm1_beta = kwargs.get("gemm1_beta", None)
        gemm1_clamp_limit = kwargs.get("gemm1_clamp_limit", None)

        # BF16 hidden states
        hidden_states_bf16 = hidden_states_orig.to(torch.bfloat16)

        with autotune(enable_autotune):
            output = trtllm_fp4_block_scale_moe(
                routing_logits=expert_logits,
                routing_bias=routing_bias,
                hidden_states=hidden_states_bf16,
                hidden_states_scale=None,  # BF16 doesn't need scale
                gemm1_weights=static_data["gemm1_weights_fp4_shuffled"],
                gemm1_weights_scale=static_data["gemm1_scales_fp4_shuffled"],
                gemm1_bias=gemm1_bias,
                gemm1_alpha=gemm1_alpha,
                gemm1_beta=gemm1_beta,
                gemm1_clamp_limit=gemm1_clamp_limit,
                gemm2_weights=static_data["gemm2_weights_fp4_shuffled"],
                gemm2_weights_scale=static_data["gemm2_scales_fp4_shuffled"],
                gemm2_bias=gemm2_bias,
                output1_scale_scalar=static_data["scale_c_fc1"],
                output1_scale_gate_scalar=static_data["scale_gate_fc1"],
                output2_scale_scalar=static_data["scale_c_fc2"],
                num_experts=num_experts,
                top_k=top_k,
                n_group=n_groups,
                topk_group=top_k_groups,
                intermediate_size=intermediate_size,
                local_expert_offset=0,
                local_num_experts=num_experts,
                routed_scaling_factor=routed_scaling,
                tile_tokens_dim=None,
                routing_method_type=routing_method_type,
                gated_act_type=gated_act_type,
                do_finalize=True,
                tune_max_num_tokens=TUNE_MAX_NUM_TOKENS,
            )
        # Handle different return types (tuple, list, or tensor)
        if isinstance(output, (tuple, list)):
            return output[0].to(torch.float)
        return output.to(torch.float)

    def compute_reference(self, args):
        """Compute reference output using dequantized operations."""
        return run_moe_reference_fp4_bf16(args, self.quant_mode)

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
    gated_act_type: GatedActType = GatedActType.SwiGlu,
    routing_method_type: RoutingMethodType = RoutingMethodType.Renormalize,
    n_groups: int | None = None,
    top_k_groups: int | None = None,
    routed_scaling: float | None = None,
    has_routing_bias: bool = False,
    use_gemm_bias: bool = False,
    gemm1_alpha_value: float | None = None,
    gemm1_beta_value: float | None = None,
    gemm1_clamp_limit_value: float | None = None,
    seed: int = 0,
):
    """
    Test MxFP4 x BF16 fused MoE against reference implementation.

    Args:
        num_tokens: Number of input tokens
        hidden_size: Hidden dimension size
        intermediate_size: Intermediate (FFN) dimension size
        num_experts: Number of experts in MoE
        top_k: Number of experts selected per token
        padding: Padding for expert token counts
        gated_act_type: Type of gated activation (SwiGlu or GeGlu)
        routing_method_type: Routing method type
        n_groups: Number of groups for grouped routing
        top_k_groups: Number of top-k groups
        routed_scaling: Scaling factor for routed outputs
        has_routing_bias: Whether to use routing bias
        use_gemm_bias: Whether to use bias for GEMM1 and GEMM2
        gemm1_alpha_value: If provided, sets the swiglu alpha for all experts (float32)
        gemm1_beta_value: If provided, sets the linear offset (e.g., +1 in GPT OSS)
        gemm1_clamp_limit_value: If provided, sets the clamp limit for activation inputs
        seed: Random seed
    """
    print(f"\n{'='*70}")
    print(f"Testing MxFP4 x BF16 Fused MoE")
    print(f"  num_tokens={num_tokens}, hidden_size={hidden_size}")
    print(f"  intermediate_size={intermediate_size}, num_experts={num_experts}")
    print(f"  top_k={top_k}, gated_act_type={gated_act_type.name}")
    print(f"  routing_method_type={routing_method_type.name}")
    print(f"  use_gemm_bias={use_gemm_bias}, alpha={gemm1_alpha_value}")
    print(f"  beta={gemm1_beta_value}, clamp_limit={gemm1_clamp_limit_value}")
    print(f"{'='*70}")

    torch.cuda.synchronize()
    torch.random.manual_seed(seed)

    moe_impl = FP4Moe(quant_mode=QuantMode.FP4_MXFP4_Bf16)

    # Validation checks
    assert top_k <= num_experts, f"top_k ({top_k}) must be <= num_experts ({num_experts})"
    assert top_k <= 10, "top_k must be <= 10"

    # Create test data
    if routing_method_type == RoutingMethodType.DeepSeekV3:
        expert_logits = torch.randn((num_tokens, num_experts), device="cuda").to(
            torch.float
        )
    else:
        expert_logits = torch.randn((num_tokens, num_experts), device="cuda").to(
            torch.bfloat16
        )

    if has_routing_bias:
        routing_bias = torch.randn(num_experts, device="cuda", dtype=torch.bfloat16)
    else:
        routing_bias = None

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

    # Create bias tensors if enabled
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

    # Generate routing info
    use_routing_scales_on_input = routing_method_type == RoutingMethodType.Llama4

    if routing_method_type == RoutingMethodType.DeepSeekV3:
        permute_info, scores = routing_reference_no_aux(
            expert_logits,
            routing_bias,
            top_k,
            n_groups,
            top_k_groups,
            routed_scaling,
            padding,
            use_routing_scales_on_input,
        )
    elif routing_method_type == RoutingMethodType.Renormalize:
        permute_info, scores = routing_reference_renormalize(
            expert_logits, top_k, num_experts, padding
        )
    else:
        raise NotImplementedError(
            f"Routing method {routing_method_type} not implemented in this test"
        )

    # 1. Quantize weights offline
    print("Quantizing weights...")
    weights_data = moe_impl.quantize_weights(
        gemm1_weights, gemm2_weights, hidden_states
    )

    # 2. Quantize inputs at runtime
    print("Preparing inputs...")
    inputs_data = moe_impl.quantize_inputs(
        hidden_states, weights_data["hidden_states_scale_global"]
    )

    # 3. Combine quantized data
    quant_data = {**weights_data, **inputs_data}

    # Create arguments for reference computation
    args = moe_args(
        num_tokens,
        num_experts,
        hidden_size,
        intermediate_size,
        top_k,
        padding,
        quant_data["hidden_states"],
        quant_data["hidden_states_scale"],
        quant_data["hidden_states_scale_global"],
        scores,
        quant_data["gemm1_weights"],
        quant_data["gemm1_scales"],
        quant_data["gemm1_scales_global"],
        quant_data["gemm2_weights"],
        quant_data["gemm2_scales"],
        quant_data["gemm2_scales_global"],
        permute_info,
        use_routing_scales_on_input,
        gated_act_type.value,
        gemm1_bias=gemm1_bias,
        gemm2_bias=gemm2_bias,
        gemm1_alpha=gemm1_alpha,
        gemm1_beta=gemm1_beta,
        gemm1_clamp_limit=gemm1_clamp_limit,
    )

    # Compute reference output
    print("Computing reference output...")
    output_reference, args_dequant = moe_impl.compute_reference(args)

    if output_reference is None:
        raise RuntimeError("Reference computation failed to produce output")

    # Prepare static weights for kernel
    print("Preparing weights for kernel...")
    static_data = moe_impl.prepare_static_weights_for_kernel(
        args_dequant,
        args,
        gemm1_weights,
        gemm2_weights,
        hidden_size,
        intermediate_size,
        num_experts,
    )

    # Compute actual output using FlashInfer kernel
    print("Computing FlashInfer kernel output...")
    output_actual = moe_impl.call_moe(
        static_data,
        hidden_states,
        weights_data["hidden_states_scale_global"],
        expert_logits=expert_logits,
        routing_bias=routing_bias,
        num_experts=num_experts,
        top_k=top_k,
        n_groups=n_groups,
        top_k_groups=top_k_groups,
        intermediate_size=intermediate_size,
        routed_scaling=routed_scaling,
        routing_method_type=routing_method_type,
        gated_act_type=gated_act_type.value,
        gemm1_bias=gemm1_bias,
        gemm2_bias=gemm2_bias,
        gemm1_alpha=gemm1_alpha,
        gemm1_beta=gemm1_beta,
        gemm1_clamp_limit=gemm1_clamp_limit,
        enable_autotune=True,
    )

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
# Main Entry Point
# ====================================================================================


if __name__ == "__main__":
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
            gated_act_type=GatedActType.SwiGlu,
            routing_method_type=RoutingMethodType.Renormalize,
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
            gated_act_type=GatedActType.SwiGlu,
            routing_method_type=RoutingMethodType.Renormalize,
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
            gated_act_type=GatedActType.SwiGlu,
            routing_method_type=RoutingMethodType.Renormalize,
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
            gated_act_type=GatedActType.SwiGlu,
            routing_method_type=RoutingMethodType.Renormalize,
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
            gated_act_type=GatedActType.SwiGlu,
            routing_method_type=RoutingMethodType.Renormalize,
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
            gated_act_type=GatedActType.SwiGlu,
            routing_method_type=RoutingMethodType.Renormalize,
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
            gated_act_type=GatedActType.SwiGlu,
            routing_method_type=RoutingMethodType.Renormalize,
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
            gated_act_type=GatedActType.SwiGlu,
            routing_method_type=RoutingMethodType.Renormalize,
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
            gated_act_type=GatedActType.SwiGlu,
            routing_method_type=RoutingMethodType.Renormalize,
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
