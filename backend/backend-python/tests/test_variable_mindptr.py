"""
Test script documenting FlashInfer's group_gemm_mxfp4_nt_groupwise scale layout requirements.

KEY FINDING:
=============
Per-group swizzling with _pad_scale_factors DOES NOT WORK for variable group sizes.

The reason is that FlashInfer expects scale layout based on CUMULATIVE m_indptr values:
    m_indptr_padded[i] = ((m_indptr[i] + i * (ALIGNMENT_M_SF - 1)) // ALIGNMENT_M_SF) * ALIGNMENT_M_SF

This means:
- Scale sizes depend on ALL previous groups, not just the current group
- _pad_scale_factors called independently for each group produces DIFFERENT offsets
- The uniform padding + _swizzle_blockscale approach correctly implements this formula

EXAMPLE:
========
For group_sizes = [148, 200, 64, 180, 96] and ALIGNMENT_M_SF = 128:

FlashInfer formula (cumulative):
  m_indptr = [0, 148, 348, 412, 592, 688]
  m_indptr_padded = [0, 256, 512, 768, 1024, 1280]
  Scale sizes: [256, 256, 256, 256, 256]

Per-group _pad_scale_factors (WRONG):
  Offsets: [0, 256, 512, 640, 896, 1024]
  Scale sizes: [256, 256, 128, 256, 128]
  
The layouts DON'T match! This causes incorrect GEMM results.

SOLUTION:
=========
Use uniform padding where all groups are padded to the same max_group_size_padded.
Then use _swizzle_blockscale which correctly implements the cumulative formula.
"""

import torch
from flashinfer.gemm import group_gemm_mxfp4_nt_groupwise
from flashinfer import mxfp8_quantize, mxfp4_quantize
from flashinfer.fp4_quantization import _pad_scale_factors

import sys
sys.path.insert(0, '/home/acciente/work/pie/backend/backend-python')
from model.gptoss import _swizzle_blockscale


def test_uniform_padding_correctness():
    """Verify uniform padding produces correct results."""
    print("=" * 60)
    print("Test: Uniform padding correctness")
    print("=" * 60)
    
    torch.manual_seed(42)
    device = 'cuda'

    # Various group sizes
    group_sizes = [4, 12, 8, 20, 4, 16, 8, 12]
    num_groups = len(group_sizes)
    cum_m = sum(group_sizes)
    k = 128
    n = 64
    k_groups = k // 32
    ALIGNMENT_M_SF = 128
    
    max_size = max(group_sizes)
    max_padded = ((max_size + 3) // 4) * 4

    # Original m_indptr
    m_indptr_orig = torch.tensor(
        [0] + list(torch.cumsum(torch.tensor(group_sizes), 0).tolist()), 
        dtype=torch.int32, device=device
    )

    # Uniform m_indptr
    m_indptr_uniform = torch.arange(
        0, (num_groups + 1) * max_padded, max_padded,
        dtype=torch.int32, device=device
    )

    # Activations
    a_bf16 = torch.ones(cum_m, k, dtype=torch.bfloat16, device=device)
    a_fp8_orig, a_sf = mxfp8_quantize(a_bf16, is_sf_swizzled_layout=False)
    a_sf_2d = a_sf.reshape(cum_m, k_groups)

    # Pad activations uniformly
    a_fp8 = torch.zeros(num_groups * max_padded, k, dtype=a_fp8_orig.dtype, device=device)
    a_scale = torch.full((num_groups * max_padded, k_groups), 127, dtype=torch.uint8, device=device)
    
    for i in range(num_groups):
        src_start = m_indptr_orig[i].item()
        src_end = m_indptr_orig[i+1].item()
        dst_start = i * max_padded
        size = src_end - src_start
        if size > 0:
            a_fp8[dst_start:dst_start + size] = a_fp8_orig[src_start:src_end]
            a_scale[dst_start:dst_start + size] = a_sf_2d[src_start:src_end]

    # Weights
    b_bf16 = torch.ones(num_groups, n, k, dtype=torch.bfloat16, device=device)
    b_fp4, b_sf = mxfp4_quantize(b_bf16)
    b_sf_3d = b_sf[:num_groups * n].view(num_groups, n, k_groups)
    b_sf_swizzled = _swizzle_blockscale(b_sf_3d, num_groups, n, k, 32)

    # Swizzle a_scale using _swizzle_blockscale (correct approach)
    a_scale_3d = a_scale.reshape(num_groups, max_padded, k_groups)
    a_scale_swizzled = _swizzle_blockscale(
        a_scale_3d, num_groups, max_padded, k, 32
    ).flatten(0, 1)

    # Apply ALIGNMENT_M_SF padding
    group_arange = torch.arange(0, num_groups + 1, dtype=torch.int32, device=device)
    m_indptr_for_sf = group_arange * max_padded
    m_indptr_padded = (
        (m_indptr_for_sf + group_arange * (ALIGNMENT_M_SF - 1))
        // ALIGNMENT_M_SF * ALIGNMENT_M_SF
    )
    m_sf = m_indptr_padded[1:] - m_indptr_padded[:-1]
    
    a_scale_chunks = a_scale_swizzled.chunk(num_groups, dim=0)
    a_scale_final = torch.cat([
        torch.cat([
            chunk,
            torch.zeros(
                m_sf[i].item() - chunk.shape[0],
                chunk.shape[1],
                dtype=chunk.dtype,
                device=chunk.device,
            ),
        ])
        for i, chunk in enumerate(a_scale_chunks)
    ])

    # GEMM
    output_padded = group_gemm_mxfp4_nt_groupwise(
        a_fp8, b_fp4, a_scale_final, b_sf_swizzled, m_indptr_uniform,
        out_dtype=torch.bfloat16
    )

    # Extract results
    output = torch.zeros(cum_m, n, dtype=torch.bfloat16, device=device)
    for i in range(num_groups):
        orig_start = m_indptr_orig[i].item()
        orig_end = m_indptr_orig[i+1].item()
        size = orig_end - orig_start
        if size > 0:
            uniform_start = i * max_padded
            output[orig_start:orig_end] = output_padded[uniform_start:uniform_start + size]

    # Verify
    expected = float(k)
    has_nan = torch.isnan(output).any().item()
    has_inf = torch.isinf(output).any().item()
    
    print(f"Group sizes: {group_sizes}")
    print(f"Max padded: {max_padded}")
    print(f"NaN: {has_nan}, Inf: {has_inf}")
    
    all_correct = True
    for i in range(num_groups):
        start = m_indptr_orig[i].item()
        actual = output[start, 0].item()
        correct = abs(actual - expected) < 1.0
        if not correct:
            print(f"  Group {i}: expected {expected}, got {actual}")
            all_correct = False
    
    passed = all_correct and not has_nan and not has_inf
    print(f"Result: {'PASSED ✓' if passed else 'FAILED ✗'}")
    return passed


def test_scale_layout_formula():
    """Demonstrate the scale layout formula difference."""
    print("\n" + "=" * 60)
    print("Test: Scale layout formula comparison")
    print("=" * 60)
    
    ALIGNMENT_M_SF = 128
    
    # Test case with large groups
    group_sizes = [148, 200, 64, 180, 96]
    num_groups = len(group_sizes)
    
    m_indptr = [0] + list(torch.cumsum(torch.tensor(group_sizes), 0).tolist())
    print(f"Group sizes: {group_sizes}")
    print(f"m_indptr: {m_indptr}")
    
    # FlashInfer formula (cumulative)
    flashinfer_offsets = []
    for i, m in enumerate(m_indptr):
        offset = ((m + i * (ALIGNMENT_M_SF - 1)) // ALIGNMENT_M_SF) * ALIGNMENT_M_SF
        flashinfer_offsets.append(offset)
    
    print(f"\nFlashInfer formula (cumulative):")
    print(f"  Offsets: {flashinfer_offsets}")
    flashinfer_sizes = [flashinfer_offsets[i+1] - flashinfer_offsets[i] for i in range(num_groups)]
    print(f"  Scale sizes: {flashinfer_sizes}")
    
    # Per-group _pad_scale_factors
    pergroup_offsets = [0]
    cumulative = 0
    for size in group_sizes:
        padded = ((size + ALIGNMENT_M_SF - 1) // ALIGNMENT_M_SF) * ALIGNMENT_M_SF
        cumulative += padded
        pergroup_offsets.append(cumulative)
    
    print(f"\nPer-group _pad_scale_factors:")
    print(f"  Offsets: {pergroup_offsets}")
    pergroup_sizes = [pergroup_offsets[i+1] - pergroup_offsets[i] for i in range(num_groups)]
    print(f"  Scale sizes: {pergroup_sizes}")
    
    # Compare
    match = flashinfer_offsets == pergroup_offsets
    print(f"\nLayouts match: {match}")
    
    if not match:
        print("\nDifferences:")
        for i in range(len(flashinfer_offsets)):
            if flashinfer_offsets[i] != pergroup_offsets[i]:
                print(f"  Position {i}: FlashInfer={flashinfer_offsets[i]}, PerGroup={pergroup_offsets[i]}")
    
    return match


def main():
    print("FlashInfer Scale Layout Tests")
    print("=" * 60 + "\n")
    
    results = []
    results.append(("Uniform padding", test_uniform_padding_correctness()))
    
    print("\n" + "=" * 60)
    test_scale_layout_formula()
    
    print("\n" + "=" * 60)
    print("CONCLUSION")
    print("=" * 60)
    print("""
The uniform padding approach is CORRECT because:
1. All groups are padded to the same max_group_size_padded
2. _swizzle_blockscale correctly implements the cumulative formula
3. The ALIGNMENT_M_SF padding after swizzle matches FlashInfer's expectation

Per-group swizzling FAILS because:
1. _pad_scale_factors pads each group independently to multiple of 128
2. This produces different cumulative offsets than FlashInfer expects
3. The GEMM kernel reads scales from wrong positions, causing incorrect results
""")
    
    return results[0][1]


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
