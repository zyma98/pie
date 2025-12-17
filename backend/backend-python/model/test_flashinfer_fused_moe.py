"""
Sanity check script for FlashInfer's cutlass_fused_moe function with
bfloat16 activations and MXFP4 expert weights.

This test verifies our understanding of the FlashInfer fused MoE API
before integrating it into the GPT OSS model.

Reference: https://github.com/flashinfer-ai/flashinfer/blob/main/tests/moe/test_trtllm_cutlass_fused_moe.py
"""

import torch
from torch.nn import functional as F

import flashinfer.fused_moe as fused_moe
from flashinfer import mxfp4_dequantize_host


# ============================================================================
# GPT OSS-like parameters
# ============================================================================
# These match the SwiGLU activation in GPT OSS:
#   x_glu = x_glu.clamp(min=None, max=swiglu_limit)
#   x_linear = x_linear.clamp(min=-swiglu_limit, max=swiglu_limit)
#   out_glu = x_glu * torch.sigmoid(1.702 * x_glu)
#   t = out_glu * (x_linear + 1)
SWIGLU_ALPHA = 1.702
SWIGLU_BETA = 1.0
SWIGLU_LIMIT = 7.0

# Test configuration (small sizes for quick sanity check)
BATCH_SIZE = 4
HIDDEN_SIZE = 256  # Must be divisible by 32 for MXFP4
INTERMEDIATE_SIZE = 512  # Must be divisible by 32 for MXFP4
NUM_EXPERTS = 4
TOP_K = 2


def compute_routing(
    router_logits: torch.Tensor, top_k: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute routing weights and selected experts from router logits.

    Args:
        router_logits: Router logits of shape [batch_size, num_experts]
        top_k: Number of experts to route to per token

    Returns:
        tuple containing:
            - routing_weights: Expert weights of shape [batch_size, top_k]
            - selected_experts: Expert indices of shape [batch_size, top_k]
    """
    routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
    routing_weights, selected_experts = torch.topk(routing_weights, top_k, dim=-1)
    routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
    routing_weights = routing_weights.float()
    return routing_weights, selected_experts


def dequant_mxfp4_batches_host(
    mat_fp4: torch.Tensor,
    scale_tensor: torch.Tensor,
) -> torch.Tensor:
    """
    Dequantize MXFP4 weights for multiple experts (batched).

    Args:
        mat_fp4: Quantized weights of shape [num_experts, out_dim, in_dim//2]
        scale_tensor: Scales of shape [num_experts, out_dim, in_dim//32]

    Returns:
        Dequantized weights of shape [num_experts, out_dim, in_dim]
    """
    return torch.stack(
        [
            mxfp4_dequantize_host(mat_fp4[b, :, :], scale_tensor[b, :, :])
            for b in range(mat_fp4.size(0))
        ]
    )


def compute_with_experts_reference(
    num_experts: int,
    x: torch.Tensor,
    w31_weight: torch.Tensor,
    w2_weight: torch.Tensor,
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
    alpha: float | None = None,
    beta: float | None = None,
    limit: float | None = None,
) -> torch.Tensor:
    """
    Reference implementation for MoE forward pass.

    This matches the GPT OSS SwiGLU activation when alpha, beta, and limit are provided.

    Args:
        num_experts: Number of experts
        x: Input tensor of shape [batch_size, hidden_size]
        w31_weight: Gate+Up projection weights [num_experts, 2*intermediate_size, hidden_size]
        w2_weight: Down projection weights [num_experts, hidden_size, intermediate_size]
        selected_experts: Selected expert indices [batch_size, top_k]
        routing_weights: Expert routing weights [batch_size, top_k]
        alpha: SwiGLU alpha parameter (sigmoid coefficient)
        beta: SwiGLU beta parameter (bias added to linear path)
        limit: SwiGLU limit parameter (clamping value)

    Returns:
        Output tensor of shape [batch_size, hidden_size]
    """
    results = torch.zeros_like(x)

    for expert_id in range(num_experts):
        mask = selected_experts == expert_id
        if not mask.sum():
            continue
        batch_idx, nth_expert = torch.where(mask)

        # Get expert weights
        w31_expert = w31_weight[expert_id]  # [2 * intermediate_size, hidden_size]
        w2_expert = w2_weight[expert_id]  # [hidden_size, intermediate_size]

        # Split w31 into w3 (up) and w1 (gate)
        # Note: FlashInfer expects [w3, w1] concatenation order
        w3_expert, w1_expert = torch.chunk(w31_expert, 2, dim=0)

        expert_inputs = x[batch_idx]

        if alpha is not None and limit is not None and beta is not None:
            # SwiGLUBias activation (matches GPT OSS)
            # Gate path (w1): x_glu = clamp(x @ w1.T, max=limit)
            x1 = expert_inputs @ w1_expert.t()
            x1 = x1.clamp_(min=None, max=limit)
            x1_scaled = x1 * torch.sigmoid(alpha * x1)

            # Up path (w3): x_linear = clamp(x @ w3.T, min=-limit, max=limit) + beta
            x2 = expert_inputs @ w3_expert.t()
            x2 = x2.clamp_(min=-limit, max=limit) + beta

            inter = x1_scaled * x2
        else:
            # Standard SwiGLU (no clamping or bias)
            inter = F.silu(expert_inputs @ w1_expert.t()) * (
                expert_inputs @ w3_expert.t()
            )

        # Down projection
        output = inter @ w2_expert.t()
        results[batch_idx] += routing_weights[batch_idx, nth_expert, None] * output

    return results.view_as(x)


def test_moe_bf16_mxfp4():
    """
    Test FlashInfer's cutlass_fused_moe with bfloat16 activations and MXFP4 weights.

    This test uses GPT OSS-like SwiGLU parameters (alpha=1.702, beta=1.0, limit=7.0).
    """
    torch.manual_seed(42)

    # Dimensions
    e = NUM_EXPERTS
    m = BATCH_SIZE
    n = INTERMEDIATE_SIZE
    k = HIDDEN_SIZE

    print(f"Testing fused MoE with:")
    print(f"  batch_size={m}, hidden_size={k}, intermediate_size={n}")
    print(f"  num_experts={e}, top_k={TOP_K}")
    print(f"  SwiGLU: alpha={SWIGLU_ALPHA}, beta={SWIGLU_BETA}, limit={SWIGLU_LIMIT}")
    print()

    # Create input tensor (bfloat16)
    x = torch.randn(m, k, dtype=torch.bfloat16).cuda()

    # Create random MXFP4 weights
    # MXFP4 format: 2 values packed into 1 byte, so shape is [out_dim, in_dim//2]
    # Scale shape: [out_dim, in_dim//32] (group size of 32)
    w1 = torch.randint(0, 256, (e, 2 * n, k // 2), device="cuda", dtype=torch.uint8)
    w2 = torch.randint(0, 256, (e, k, n // 2), device="cuda", dtype=torch.uint8)

    # Scales in E8M0 format (stored as uint8)
    # Use reasonable scale values (around 118-123 corresponds to scales near 1.0)
    w1_scale = torch.randint(
        118, 123, (e, 2 * n, k // 32), device="cuda", dtype=torch.uint8
    )
    w2_scale = torch.randint(
        118, 123, (e, k, n // 32), device="cuda", dtype=torch.uint8
    )

    # Create router logits and compute routing
    router_logits = torch.randn(m, e, dtype=torch.bfloat16).cuda()
    routing_weights, selected_experts = compute_routing(router_logits, TOP_K)

    # Prepare SwiGLU parameters (per-expert tensors)
    alpha_t = torch.ones(e, device="cuda") * SWIGLU_ALPHA
    limit_t = torch.ones(e, device="cuda") * SWIGLU_LIMIT
    beta_t = torch.ones(e, device="cuda") * SWIGLU_BETA

    # Prepare quantization scales for FlashInfer
    # Format: [w1_scale.view(int32), w2_scale.view(int32)]
    quant_scales = [
        w1_scale.view(torch.int32),
        w2_scale.view(torch.int32),
    ]

    # FlashInfer requires hidden_size to be padded to multiple of 32
    # (already satisfied in our test)
    pad_size = k - x.shape[1]
    x_pad = torch.nn.functional.pad(x, (0, pad_size)) if pad_size > 0 else x

    # Run FlashInfer fused MoE
    flash_output = torch.zeros_like(x)
    _ = fused_moe.cutlass_fused_moe(
        x_pad,
        selected_experts.to(torch.int),
        routing_weights,
        w1.contiguous().view(torch.uint8),
        w2.contiguous().view(torch.uint8),
        torch.bfloat16,
        swiglu_alpha=alpha_t,
        swiglu_limit=limit_t,
        swiglu_beta=beta_t,
        quant_scales=quant_scales,
        use_w4_group_scaling=True,
        output=flash_output,
    )

    # Dequantize weights for reference computation
    dq_mxfp4_w1 = (
        dequant_mxfp4_batches_host(
            w1.cpu(),
            w1_scale.cpu(),
        )
        .cuda()
        .to(torch.bfloat16)
    )

    dq_mxfp4_w2 = (
        dequant_mxfp4_batches_host(
            w2.cpu(),
            w2_scale.cpu(),
        )
        .cuda()
        .to(torch.bfloat16)
    )

    # Run reference implementation
    ref_output = compute_with_experts_reference(
        e,
        x,
        dq_mxfp4_w1,
        dq_mxfp4_w2,
        selected_experts,
        routing_weights,
        SWIGLU_ALPHA,
        SWIGLU_BETA,
        SWIGLU_LIMIT,
    )

    # Compare outputs
    try:
        torch.testing.assert_close(ref_output, flash_output, rtol=1e-1, atol=1e-1)
        print("✓ Test PASSED!")
        print(f"  Max absolute difference: {(ref_output - flash_output).abs().max().item():.6f}")
        print(f"  Mean absolute difference: {(ref_output - flash_output).abs().mean().item():.6f}")
    except AssertionError as e:
        print("✗ Test FAILED!")
        print(f"  Max absolute difference: {(ref_output - flash_output).abs().max().item():.6f}")
        print(f"  Mean absolute difference: {(ref_output - flash_output).abs().mean().item():.6f}")
        raise e


def test_moe_bf16_mxfp4_no_swiglu_bias():
    """
    Additional test without SwiGLU bias (alpha, beta, limit = None).

    This tests standard SwiGLU activation for comparison.
    """
    torch.manual_seed(42)

    e = NUM_EXPERTS
    m = BATCH_SIZE
    n = INTERMEDIATE_SIZE
    k = HIDDEN_SIZE

    print(f"\nTesting fused MoE with standard SwiGLU (no bias):")
    print(f"  batch_size={m}, hidden_size={k}, intermediate_size={n}")
    print(f"  num_experts={e}, top_k={TOP_K}")
    print()

    x = torch.randn(m, k, dtype=torch.bfloat16).cuda()
    w1 = torch.randint(0, 256, (e, 2 * n, k // 2), device="cuda", dtype=torch.uint8)
    w2 = torch.randint(0, 256, (e, k, n // 2), device="cuda", dtype=torch.uint8)
    w1_scale = torch.randint(
        118, 123, (e, 2 * n, k // 32), device="cuda", dtype=torch.uint8
    )
    w2_scale = torch.randint(
        118, 123, (e, k, n // 32), device="cuda", dtype=torch.uint8
    )

    router_logits = torch.randn(m, e, dtype=torch.bfloat16).cuda()
    routing_weights, selected_experts = compute_routing(router_logits, TOP_K)

    quant_scales = [
        w1_scale.view(torch.int32),
        w2_scale.view(torch.int32),
    ]

    flash_output = torch.zeros_like(x)
    _ = fused_moe.cutlass_fused_moe(
        x,
        selected_experts.to(torch.int),
        routing_weights,
        w1.contiguous().view(torch.uint8),
        w2.contiguous().view(torch.uint8),
        torch.bfloat16,
        swiglu_alpha=None,
        swiglu_limit=None,
        swiglu_beta=None,
        quant_scales=quant_scales,
        use_w4_group_scaling=True,
        output=flash_output,
    )

    # Dequantize weights for reference computation
    dq_mxfp4_w1 = (
        dequant_mxfp4_batches_host(w1.cpu(), w1_scale.cpu())
        .cuda()
        .to(torch.bfloat16)
    )
    dq_mxfp4_w2 = (
        dequant_mxfp4_batches_host(w2.cpu(), w2_scale.cpu())
        .cuda()
        .to(torch.bfloat16)
    )

    ref_output = compute_with_experts_reference(
        e,
        x,
        dq_mxfp4_w1,
        dq_mxfp4_w2,
        selected_experts,
        routing_weights,
        alpha=None,
        beta=None,
        limit=None,
    )

    try:
        torch.testing.assert_close(ref_output, flash_output, rtol=1e-1, atol=1e-1)
        print("✓ Test PASSED!")
        print(f"  Max absolute difference: {(ref_output - flash_output).abs().max().item():.6f}")
        print(f"  Mean absolute difference: {(ref_output - flash_output).abs().mean().item():.6f}")
    except AssertionError as e:
        print("✗ Test FAILED!")
        print(f"  Max absolute difference: {(ref_output - flash_output).abs().max().item():.6f}")
        print(f"  Mean absolute difference: {(ref_output - flash_output).abs().mean().item():.6f}")
        raise e


def test_gptoss_like_dimensions():
    """
    Test with dimensions closer to actual GPT OSS model configuration.
    """
    torch.manual_seed(42)

    # GPT OSS-like dimensions (scaled down for testing)
    e = 8  # num_experts
    m = 16  # batch_size
    n = 1024  # intermediate_size (actual might be larger)
    k = 512  # hidden_size (actual might be larger)
    top_k = 2

    print(f"\nTesting fused MoE with GPT OSS-like dimensions:")
    print(f"  batch_size={m}, hidden_size={k}, intermediate_size={n}")
    print(f"  num_experts={e}, top_k={top_k}")
    print(f"  SwiGLU: alpha={SWIGLU_ALPHA}, beta={SWIGLU_BETA}, limit={SWIGLU_LIMIT}")
    print()

    x = torch.randn(m, k, dtype=torch.bfloat16).cuda()
    w1 = torch.randint(0, 256, (e, 2 * n, k // 2), device="cuda", dtype=torch.uint8)
    w2 = torch.randint(0, 256, (e, k, n // 2), device="cuda", dtype=torch.uint8)
    w1_scale = torch.randint(
        118, 123, (e, 2 * n, k // 32), device="cuda", dtype=torch.uint8
    )
    w2_scale = torch.randint(
        118, 123, (e, k, n // 32), device="cuda", dtype=torch.uint8
    )

    router_logits = torch.randn(m, e, dtype=torch.bfloat16).cuda()
    routing_weights, selected_experts = compute_routing(router_logits, top_k)

    alpha_t = torch.ones(e, device="cuda") * SWIGLU_ALPHA
    limit_t = torch.ones(e, device="cuda") * SWIGLU_LIMIT
    beta_t = torch.ones(e, device="cuda") * SWIGLU_BETA

    quant_scales = [
        w1_scale.view(torch.int32),
        w2_scale.view(torch.int32),
    ]

    flash_output = torch.zeros_like(x)
    _ = fused_moe.cutlass_fused_moe(
        x,
        selected_experts.to(torch.int),
        routing_weights,
        w1.contiguous().view(torch.uint8),
        w2.contiguous().view(torch.uint8),
        torch.bfloat16,
        swiglu_alpha=alpha_t,
        swiglu_limit=limit_t,
        swiglu_beta=beta_t,
        quant_scales=quant_scales,
        use_w4_group_scaling=True,
        output=flash_output,
    )

    dq_mxfp4_w1 = (
        dequant_mxfp4_batches_host(w1.cpu(), w1_scale.cpu())
        .cuda()
        .to(torch.bfloat16)
    )
    dq_mxfp4_w2 = (
        dequant_mxfp4_batches_host(w2.cpu(), w2_scale.cpu())
        .cuda()
        .to(torch.bfloat16)
    )

    ref_output = compute_with_experts_reference(
        e,
        x,
        dq_mxfp4_w1,
        dq_mxfp4_w2,
        selected_experts,
        routing_weights,
        SWIGLU_ALPHA,
        SWIGLU_BETA,
        SWIGLU_LIMIT,
    )

    try:
        torch.testing.assert_close(ref_output, flash_output, rtol=1e-1, atol=1e-1)
        print("✓ Test PASSED!")
        print(f"  Max absolute difference: {(ref_output - flash_output).abs().max().item():.6f}")
        print(f"  Mean absolute difference: {(ref_output - flash_output).abs().mean().item():.6f}")
    except AssertionError as e:
        print("✗ Test FAILED!")
        print(f"  Max absolute difference: {(ref_output - flash_output).abs().max().item():.6f}")
        print(f"  Mean absolute difference: {(ref_output - flash_output).abs().mean().item():.6f}")
        raise e


if __name__ == "__main__":
    print("=" * 70)
    print("FlashInfer Fused MoE Sanity Check")
    print("Testing: BF16 activations + MXFP4 weights")
    print("=" * 70)
    print()

    # Check GPU capability
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        exit(1)

    sm_major = torch.cuda.get_device_capability()[0]
    sm_minor = torch.cuda.get_device_capability()[1]
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Compute Capability: SM{sm_major}{sm_minor}")
    print()

    if sm_major != 9:
        print(f"WARNING: BF16xMXFP4 is only supported on SM90 (Hopper).")
        print(f"         Current GPU is SM{sm_major}{sm_minor}.")
        print("         Tests may fail or be skipped.")
        print()

    try:
        test_moe_bf16_mxfp4()
        test_moe_bf16_mxfp4_no_swiglu_bias()
        test_gptoss_like_dimensions()
        print("\n" + "=" * 70)
        print("All tests PASSED!")
        print("=" * 70)
    except Exception as e:
        print("\n" + "=" * 70)
        print(f"Test failed with error: {e}")
        print("=" * 70)
        exit(1)
