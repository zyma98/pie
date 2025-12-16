# MXFP4 Format Guide for FlashInfer Integration

This document provides a comprehensive guide to the MXFP4 (Microscaling FP4) format, its usage in the GPT OSS model, and the specific requirements for FlashInfer's `group_gemm_mxfp4_nt_groupwise` function.

## Table of Contents

1. [Understanding the Dimensions](#understanding-the-dimensions)
2. [What is MXFP4?](#what-is-mxfp4)
3. [FP4 E2M1 Encoding](#fp4-e2m1-encoding)
4. [Block Scaling (UE8M0 Format)](#block-scaling-ue8m0-format)
5. [Nibble Packing](#nibble-packing)
6. [Current GPT OSS Format](#current-gpt-oss-format)
7. [FlashInfer Expected Format](#flashinfer-expected-format)
8. [Scale Swizzling](#scale-swizzling)
9. [Group GEMM Requirements](#group-gemm-requirements)
10. [Format Conversion Summary](#format-conversion-summary)
11. [Understanding `swap_ab`](#understanding-swap_ab)
12. [Column-Major vs Row-Major Scale Layouts](#column-major-vs-row-major-scale-layouts)
13. [Runtime Activation Quantization](#runtime-activation-quantization-bfloat16--mxfp8)

---

## Understanding the Dimensions

Before diving into formats, let's clarify the dimension naming used in FlashInfer's `group_gemm_mxfp4_nt_groupwise`.

### Standard GEMM Notation

In standard matrix multiplication `C = A @ B`:

```
A: (M, K)  ×  B: (K, N)  =  C: (M, N)
     │  │        │  │           │  │
     │  └────────┘  │           │  │
     │   (contract) │           │  │
     └──────────────┼───────────┘  │
        (rows)      └──────────────┘
                       (cols)
```

### FlashInfer's `group_gemm_mxfp4_nt_groupwise` Notation

This function computes **C = A @ B.T** (note the transpose!) for multiple groups:

```
A: (M, K)  ×  B.T: (K, N)  =  C: (M, N)
                    ↑
            B stored as (N, K), transposed during compute
```

### Dimension Definitions

| Symbol | Name | Description | MoE Context |
|--------|------|-------------|-------------|
| **M** | Rows | Number of input rows (tokens) | Total tokens processed |
| **K** | Inner | Shared/contracted dimension | `hidden_size` (input features) |
| **N** | Cols | Output columns (features) | `intermediate_size` (output features) |
| **batch_size** | Groups | Number of weight matrices | Number of experts selected |

### The "cum_m" Concept (Cumulative M)

In **group GEMM**, different groups can have different numbers of rows. `cum_m` is the **total** number of rows across all groups:

```
Group 0: 50 tokens  ─┐
Group 1: 30 tokens   ├─► cum_m = 50 + 30 + 40 = 120 total rows
Group 2: 40 tokens  ─┘
```

The `m_indptr` array stores **cumulative boundaries**:

```
m_indptr = [0, 50, 80, 120]
            │   │   │    │
            │   │   │    └── End of group 2 (= cum_m)
            │   │   └── Start of group 2 / End of group 1
            │   └── Start of group 1 / End of group 0
            └── Start of group 0
```

### Concrete MoE Example

Let's say we're processing a batch with:
- **8 tokens** total
- **2 experts selected** (experts #3 and #7)
- Token distribution: 5 tokens → expert #3, 3 tokens → expert #7
- `hidden_size = 2048` (K)
- `intermediate_size = 8192` (N)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          MoE Forward Pass                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Activations A                    Weights B (per expert)                │
│  ┌─────────────────┐              ┌─────────────────┐                   │
│  │                 │              │ Expert #3       │                   │
│  │  Token 0  ──────┼──────────────│ (8192, 2048)    │                   │
│  │  Token 1  ──────┤   Group 0    │                 │                   │
│  │  Token 2  ──────┤   (5 tokens) └─────────────────┘                   │
│  │  Token 3  ──────┤                                                    │
│  │  Token 4  ──────┘              ┌─────────────────┐                   │
│  │                                │ Expert #7       │                   │
│  │  Token 5  ──────┬──────────────│ (8192, 2048)    │                   │
│  │  Token 6  ──────┤   Group 1    │                 │                   │
│  │  Token 7  ──────┘   (3 tokens) └─────────────────┘                   │
│  │                 │                                                    │
│  └─────────────────┘                                                    │
│   Shape: (8, 2048)                Shape: (2, 8192, 2048)                │
│          (cum_m, K)                      (batch_size, N, K)             │
│                                                                         │
│  m_indptr = [0, 5, 8]                                                   │
│              │  │  │                                                    │
│              │  │  └── cum_m = 8                                        │
│              │  └── Group 1 starts at row 5                             │
│              └── Group 0 starts at row 0                                │
│                                                                         │
│  Output C: (8, 8192) = (cum_m, N)                                       │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Visual: What the GEMM Computes

```
                    K=2048
              ┌───────────────┐
              │               │
         M=8  │       A       │
    (cum_m)   │   (tokens)    │
              │               │
              └───────────────┘
                      ×
              ┌───────────────┐
              │    B[0].T     │ ← Expert #3 weights, transposed
         K    │   (2048→)     │
              └───────────────┘
                   N=8192
                      ↓
              ┌───────────────┐
              │   C[0:5, :]   │ ← Output for tokens 0-4
         5    │               │
              └───────────────┘

                      ×
              ┌───────────────┐
              │    B[1].T     │ ← Expert #7 weights, transposed
         K    │   (2048→)     │
              └───────────────┘
                   N=8192
                      ↓
              ┌───────────────┐
              │   C[5:8, :]   │ ← Output for tokens 5-7
         3    │               │
              └───────────────┘
```

### Summary of Shapes

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Shape Summary                                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Input Activations (A):                                              │
│    Logical: (cum_m, K) = (8, 2048)                                   │
│    Meaning: 8 tokens, each with 2048 features                        │
│                                                                      │
│  Weights (B):                                                        │
│    Logical: (batch_size, N, K) = (2, 8192, 2048)                     │
│    Meaning: 2 experts, each mapping 2048 → 8192 features             │
│    Note: Stored as (N, K), transposed during compute                 │
│                                                                      │
│  Output (C):                                                         │
│    Logical: (cum_m, N) = (8, 8192)                                   │
│    Meaning: 8 tokens, each with 8192 output features                 │
│                                                                      │
│  m_indptr:                                                           │
│    Shape: (batch_size + 1,) = (3,)                                   │
│    Values: [0, 5, 8] — boundaries between groups                     │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Padded Dimensions

FlashInfer requires aligned dimensions:

| Original | Padded Symbol | Alignment | Example |
|----------|---------------|-----------|---------|
| `cum_m = 8` | `cum_m_padded` | Multiple of 4 | 8 (already aligned) |
| `K = 2048` | `k_padded` | Multiple of 128 | 2048 (already aligned) |
| `N = 8192` | `n_padded` | Multiple of 8 | 8192 (already aligned) |

Scale dimensions:
- `k_groups = k_padded // 32` (one scale per 32 elements)
- For K=2048: `k_groups = 2048 // 32 = 64`

---

## What is MXFP4?

**MXFP4** (Microscaling FP4) is a low-precision floating-point format designed for efficient neural network inference. It combines:

1. **4-bit floating-point values** (FP4 E2M1 format)
2. **Shared block scales** (per group of 32 values)

This achieves ~8x memory compression compared to FP32 while maintaining reasonable accuracy for inference.

### Benefits

- **Memory efficiency**: 4 bits per value + amortized scale overhead
- **Compute efficiency**: Native hardware support on NVIDIA Blackwell GPUs
- **Reasonable accuracy**: Block scaling preserves dynamic range within groups

---

## FP4 E2M1 Encoding

FP4 E2M1 uses **4 bits** per value:
- **1 sign bit** (S)
- **2 exponent bits** (E)
- **1 mantissa bit** (M)

### Bit Layout

```
┌───┬───┬───┬───┐
│ S │ E │ E │ M │
└───┴───┴───┴───┘
 3   2   1   0    (bit positions)
```

### Value Mapping

The 16 possible FP4 values map to these floating-point numbers:

| Index | Binary | Sign | Exponent | Mantissa | Value |
|-------|--------|------|----------|----------|-------|
| 0     | 0000   | +    | 00       | 0        | +0.0  |
| 1     | 0001   | +    | 00       | 1        | +0.5  |
| 2     | 0010   | +    | 01       | 0        | +1.0  |
| 3     | 0011   | +    | 01       | 1        | +1.5  |
| 4     | 0100   | +    | 10       | 0        | +2.0  |
| 5     | 0101   | +    | 10       | 1        | +3.0  |
| 6     | 0110   | +    | 11       | 0        | +4.0  |
| 7     | 0111   | +    | 11       | 1        | +6.0  |
| 8     | 1000   | -    | 00       | 0        | -0.0  |
| 9     | 1001   | -    | 00       | 1        | -0.5  |
| 10    | 1010   | -    | 01       | 0        | -1.0  |
| 11    | 1011   | -    | 01       | 1        | -1.5  |
| 12    | 1100   | -    | 10       | 0        | -2.0  |
| 13    | 1101   | -    | 10       | 1        | -3.0  |
| 14    | 1110   | -    | 11       | 0        | -4.0  |
| 15    | 1111   | -    | 11       | 1        | -6.0  |

### Lookup Table (FP4_VALUES)

```python
FP4_VALUES = (
    +0.0, +0.5, +1.0, +1.5, +2.0, +3.0, +4.0, +6.0,  # indices 0-7
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,  # indices 8-15
)
```

---

## Block Scaling (UE8M0 Format)

Since FP4 has very limited dynamic range (0 to ±6), MXFP4 uses **per-block scaling** to extend the representable range.

### Scale Format: UE8M0

Scales are stored as **unsigned 8-bit integers** representing the **log2 exponent with bias 127**:

```
actual_scale = 2^(scale_uint8 - 127)
```

### Example Scale Values

| scale_uint8 | Exponent (scale - 127) | Actual Scale |
|-------------|------------------------|--------------|
| 127         | 0                      | 1.0          |
| 128         | 1                      | 2.0          |
| 126         | -1                     | 0.5          |
| 135         | 8                      | 256.0        |
| 119         | -8                     | 0.00390625   |

### Block Size

MXFP4 uses a **block size of 32 values** per scale:

```
┌─────────────────────────────────────────────────────────────────┐
│                    32 FP4 values (16 bytes packed)              │
├─────────────────────────────────────────────────────────────────┤
│                         1 scale (1 byte)                        │
└─────────────────────────────────────────────────────────────────┘
```

### Dequantization Formula

```
dequantized_value = fp4_value × 2^(scale_uint8 - 127)
```

---

## Nibble Packing

Two 4-bit FP4 values are packed into one 8-bit `uint8`:

### FlashInfer Packing Order

FlashInfer uses the following packing convention:

```
┌─────────────┬─────────────┐
│ High Nibble │ Low Nibble  │
│  (bits 7-4) │ (bits 3-0)  │
├─────────────┼─────────────┤
│   Even idx  │   Odd idx   │
│   value[0]  │   value[1]  │
└─────────────┴─────────────┘
```

### Packing Code

```python
# Pack two FP4 values into one uint8
packed = (fp4_indices[..., 0::2] << 4) | fp4_indices[..., 1::2]
#         ↑ even indices in high nibble   ↑ odd indices in low nibble
```

### Unpacking Code

```python
# Unpack uint8 to two FP4 indices
idx_even = packed >> 4        # High nibble → even indices
idx_odd = packed & 0x0F       # Low nibble → odd indices
```

### Visual Example

Original values at positions 0-7: `[a, b, c, d, e, f, g, h]`

Packed into 4 bytes:
```
byte[0] = (a << 4) | b  → [aaaa|bbbb]
byte[1] = (c << 4) | d  → [cccc|dddd]
byte[2] = (e << 4) | f  → [eeee|ffff]
byte[3] = (g << 4) | h  → [gggg|hhhh]
```

---

## Current GPT OSS Format

The GPT OSS model stores MoE expert weights in MXFP4 format with the following structure:

### Weight Shapes

For the MoE layer with `num_experts` experts:

```python
# Gate-Up projection (fused)
gate_up_proj_blocks: (num_experts, intermediate_size * 2, hidden_size // 2)  # uint8
gate_up_proj_scales: (num_experts, intermediate_size * 2, hidden_size // 32) # uint8

# Down projection
down_proj_blocks: (num_experts, hidden_size, intermediate_size // 2)  # uint8
down_proj_scales: (num_experts, hidden_size, intermediate_size // 32) # uint8
```

### Current Processing Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                     Current GPT OSS Flow                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. Load MXFP4 weights (blocks + scales)                        │
│                     ↓                                           │
│  2. Dequantize to bfloat16 at load time                         │
│                     ↓                                           │
│  3. Store as full-precision parameters                          │
│                     ↓                                           │
│  4. Use standard torch.einsum for computation                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Dequantization Code (Current)

```python
def _dequantize_from_mxfp4(blocks, scales, fp4_values, device, dtype):
    # scales shape: (..., num_groups)
    # blocks shape: (..., num_groups, block_size)  where block_size = 16 (32 values packed)
    
    scales = scales.to(torch.int32) - 127  # Convert to exponent
    
    lut = torch.tensor(fp4_values, dtype=dtype, device=device)
    
    # Unpack nibbles
    idx_hi = (blocks >> 4).to(torch.long)    # Even indices
    idx_lo = (blocks & 0x0F).to(torch.long)  # Odd indices
    
    # Lookup and interleave
    out = torch.empty(..., block_size * 2, dtype=dtype)
    out[:, 0::2] = lut[idx_hi]  # Even positions
    out[:, 1::2] = lut[idx_lo]  # Odd positions
    
    # Apply scale
    torch.ldexp(out, scales, out=out)  # out = out * 2^scales
    
    return out
```

---

## FlashInfer Expected Format

FlashInfer's `group_gemm_mxfp4_nt_groupwise` expects specific tensor formats. Let's use a concrete example:

### Example Configuration

```python
# MoE scenario
cum_m = 120        # Total tokens across all groups
batch_size = 3     # Number of expert groups being processed
K = 2048           # Input features (hidden_size)
N = 8192           # Output features (intermediate_size)

# Token distribution across groups
# Group 0: tokens 0-49   (50 tokens)
# Group 1: tokens 50-79  (30 tokens)
# Group 2: tokens 80-119 (40 tokens)
m_indptr = [0, 50, 80, 120]

# Derived dimensions
k_padded = 2048    # Already aligned to 128
n_padded = 8192    # Already aligned to 8
k_groups = k_padded // 32  # = 64 (number of scale groups along K)
```

### Function Signature

```python
output = group_gemm_mxfp4_nt_groupwise(
    a,           # Activations: all tokens stacked
    b,           # Weights: one per expert group
    a_scale,     # Scales for activations
    b_scale,     # Scales for weights
    m_indptr,    # Tells which tokens go to which expert
    ...
)
```

### Tensor Shapes Explained

#### **a (Activations)**

```
Shape: (cum_m, k_padded) = (120, 2048)
Dtype: float8_e4m3fn

┌────────────────────────────────────────────────────────┐
│                  k_padded = 2048 features              │
│  ◄──────────────────────────────────────────────────►  │
├────────────────────────────────────────────────────────┤
│  Token 0   [f0, f1, f2, ..., f2047]                    │ ─┐
│  Token 1   [f0, f1, f2, ..., f2047]                    │  │
│  ...                                                   │  ├─ Group 0 (50 tokens)
│  Token 49  [f0, f1, f2, ..., f2047]                    │ ─┘
│  Token 50  [f0, f1, f2, ..., f2047]                    │ ─┐
│  ...                                                   │  ├─ Group 1 (30 tokens)
│  Token 79  [f0, f1, f2, ..., f2047]                    │ ─┘
│  Token 80  [f0, f1, f2, ..., f2047]                    │ ─┐
│  ...                                                   │  ├─ Group 2 (40 tokens)
│  Token 119 [f0, f1, f2, ..., f2047]                    │ ─┘
└────────────────────────────────────────────────────────┘
          cum_m = 120 rows (tokens)
```

#### **b (Weights)**

```
Shape: (batch_size, n_padded, k_padded // 2) = (3, 8192, 1024)
Dtype: uint8 (packed FP4 — two values per byte)

                    k_padded // 2 = 1024 bytes (2048 FP4 values packed)
                    ◄─────────────────────────────────────────────────►
┌───────────────────────────────────────────────────────────────────────┐
│ Expert/Group 0:                                                       │
│   ┌───────────────────────────────────────────────────────────────┐   │
│   │  Row 0 (output feat 0):    [b0, b1, ..., b1023]               │   │
│   │  Row 1 (output feat 1):    [b0, b1, ..., b1023]               │   │
│   │  ...                                                          │   │
│   │  Row 8191 (output feat N): [b0, b1, ..., b1023]               │   │
│   └───────────────────────────────────────────────────────────────┘   │
│                          n_padded = 8192 rows                         │
├───────────────────────────────────────────────────────────────────────┤
│ Expert/Group 1: (same structure)                                      │
├───────────────────────────────────────────────────────────────────────┤
│ Expert/Group 2: (same structure)                                      │
└───────────────────────────────────────────────────────────────────────┘
```

#### **a_scale (Activation Scales)**

```
Shape: (cum_m_padded_total, k_groups) = (???, 64)
Dtype: uint8 (UE8M0)
Layout: SWIZZLED + PADDED per group

Each row of activations has 64 scale values (one per 32 features).

BEFORE swizzling/padding:
┌────────────────────────────────────────┐
│ Token 0:   [s0, s1, ..., s63]          │ ─┐
│ Token 1:   [s0, s1, ..., s63]          │  │ Group 0
│ ...                                    │  │ (50 tokens)
│ Token 49:  [s0, s1, ..., s63]          │ ─┘
│ Token 50:  [s0, s1, ..., s63]          │ ─┐
│ ...                                    │  │ Group 1
│ Token 79:  [s0, s1, ..., s63]          │ ─┘ (30 tokens)
│ Token 80:  [s0, s1, ..., s63]          │ ─┐
│ ...                                    │  │ Group 2
│ Token 119: [s0, s1, ..., s63]          │ ─┘ (40 tokens)
└────────────────────────────────────────┘
Shape: (120, 64)

AFTER swizzling + per-group padding to multiple of 128:
┌────────────────────────────────────────┐
│ Group 0: 50 rows → padded to 128       │ ─┐
│   [swizzled scales...]                 │  │
│   [padding zeros...]                   │  │ 128 rows
│                                        │ ─┘
│ Group 1: 30 rows → padded to 128       │ ─┐
│   [swizzled scales...]                 │  │
│   [padding zeros...]                   │  │ 128 rows
│                                        │ ─┘
│ Group 2: 40 rows → padded to 128       │ ─┐
│   [swizzled scales...]                 │  │
│   [padding zeros...]                   │  │ 128 rows
│                                        │ ─┘
└────────────────────────────────────────┘
Final shape: (384, 64)  ← 128 × 3 groups
```

#### **b_scale (Weight Scales)**

```
Shape: (batch_size, n_padded, k_groups) = (3, 8192, 64)
Dtype: uint8 (UE8M0)
Layout: SWIZZLED

Each expert has scales for all output features:
┌─────────────────────────────────────────────────────────┐
│ Expert 0:                                               │
│   ┌─────────────────────────────────────────────────┐   │
│   │  Output 0:    [s0, s1, ..., s63]  (64 scales)   │   │
│   │  Output 1:    [s0, s1, ..., s63]                │   │
│   │  ...                                            │   │
│   │  Output 8191: [s0, s1, ..., s63]                │   │
│   └─────────────────────────────────────────────────┘   │
│                  8192 rows × 64 scale columns           │
├─────────────────────────────────────────────────────────┤
│ Expert 1: (same structure, swizzled)                    │
├─────────────────────────────────────────────────────────┤
│ Expert 2: (same structure, swizzled)                    │
└─────────────────────────────────────────────────────────┘
```

#### **m_indptr (Group Boundaries)**

```
Shape: (batch_size + 1,) = (4,)
Dtype: int32

m_indptr = [0, 50, 80, 120]
            │   │   │    │
            │   │   │    └── End of all groups (cum_m)
            │   │   │
            │   │   └── Start of group 2
            │   │       Group 2 processes tokens 80-119
            │   │       Uses weights b[2]
            │   │
            │   └── Start of group 1  
            │       Group 1 processes tokens 50-79
            │       Uses weights b[1]
            │
            └── Start of group 0
                Group 0 processes tokens 0-49
                Uses weights b[0]
```

#### **output**

```
Shape: (cum_m, n_padded) = (120, 8192)
Dtype: bfloat16 (or specified out_dtype)

┌────────────────────────────────────────────────────────┐
│                  n_padded = 8192 features              │
├────────────────────────────────────────────────────────┤
│  Token 0:   [o0, o1, ..., o8191]                       │ ─┐
│  ...                                                   │  ├─ From group 0
│  Token 49:  [o0, o1, ..., o8191]                       │ ─┘
│  Token 50:  [o0, o1, ..., o8191]                       │ ─┐
│  ...                                                   │  ├─ From group 1
│  Token 79:  [o0, o1, ..., o8191]                       │ ─┘
│  Token 80:  [o0, o1, ..., o8191]                       │ ─┐
│  ...                                                   │  ├─ From group 2
│  Token 119: [o0, o1, ..., o8191]                       │ ─┘
└────────────────────────────────────────────────────────┘
```

### Alignment Requirements

| Dimension | Alignment | Example | Notes |
|-----------|-----------|---------|-------|
| M (per group) | 4 | 50, 30, 40 → OK | Each group's token count |
| N | 8 | 8192 → OK | Output features |
| K | 128 | 2048 → OK | Input features |
| a_scale M | 128 | 50→128, 30→128, 40→128 | Per-group scale padding |

---

## Scale Swizzling

**Swizzling** is a memory layout transformation that optimizes scale factor access patterns for GPU memory hierarchy.

### Why Swizzling?

GPUs access memory in coalesced patterns. Swizzling rearranges scale factors so that threads in a warp access contiguous memory locations, maximizing memory bandwidth.

### Swizzle Pattern Visualization

For a scale tensor with shape `(m, k_groups)`:

#### Original (Row-Major) Layout

```
Row 0:  [s00, s01, s02, s03, s04, s05, s06, s07]
Row 1:  [s10, s11, s12, s13, s14, s15, s16, s17]
Row 2:  [s20, s21, s22, s23, s24, s25, s26, s27]
Row 3:  [s30, s31, s32, s33, s34, s35, s36, s37]
...
Row 31: [...]
```

Memory layout: `s00, s01, s02, ..., s07, s10, s11, ...`

#### Swizzled Layout

The swizzle interleaves scales from multiple rows so that a warp (32 threads) can load scales for different rows in one transaction:

```
Block 0: [s00, s10, s20, s30, s01, s11, s21, s31, ...]  (interleaved)
Block 1: [s40, s50, s60, s70, s41, s51, s61, s71, ...]
...
```

### Swizzle Implementation

FlashInfer provides `block_scale_interleave_sm100` for SM100 (Blackwell) architecture:

```python
def swizzle_blockscale(unswizzled_sf, batch_size, m, k_padded, tile_size=32):
    """
    Transform scale factors for optimal GPU memory access.
    
    Args:
        unswizzled_sf: Shape (batch_size, m, k_padded // tile_size)
        batch_size: Number of batches/groups
        m: M dimension
        k_padded: Padded K dimension
        tile_size: Quantization tile size (32)
    
    Returns:
        Swizzled scales with same logical shape but different memory layout
    """
    # Pad scale factors for alignment
    padded_sf = [_pad_scale_factors(unswizzled_sf[i], m, k_padded, tile_size) 
                 for i in range(batch_size)]
    padded_sf = torch.stack(padded_sf)
    
    # Apply hardware-specific interleaving
    major, minor = get_compute_capability(unswizzled_sf.device)
    swizzled = get_fp4_quantization_module(f"{major}{minor}").block_scale_interleave_sm100(
        padded_sf
    )
    
    return swizzled.view(padded_sf.shape)
```

---

## Group GEMM Requirements

`group_gemm_mxfp4_nt_groupwise` performs batched GEMM where each batch (group) can have different sizes:

### Conceptual Operation

```
For each group i:
    C[m_start:m_end, :] = A[m_start:m_end, :] @ B[i, :, :].T

Where:
    m_start = m_indptr[i]
    m_end = m_indptr[i + 1]
```

### m_indptr Structure

```python
# Example with 3 groups of sizes 128, 64, 256
m_indptr = [0, 128, 192, 448]  # cumulative boundaries
#           │   │    │    │
#           │   │    │    └── Total rows (cum_m)
#           │   │    └── Group 2 start
#           │   └── Group 1 start  
#           └── Group 0 start
```

### Critical: a_scale Padding for Group GEMM

**This is the most complex requirement!** Each group's scale factors must be padded to multiples of 128:

```python
alignment_m_sf = 128

# Calculate padded boundaries
group_arange = torch.arange(0, group_size + 1)
m_indptr_padded = (
    (m_indptr + group_arange * (alignment_m_sf - 1)) 
    // alignment_m_sf 
    * alignment_m_sf
)

# Pad each group's scales
m_sf = m_indptr_padded[1:] - m_indptr_padded[:-1]  # Size per group after padding

a_scale_chunks = a_scale_swizzled.chunk(group_size, dim=0)
a_scale_padded = [
    torch.cat([chunk, zeros_padding(m_sf[i] - chunk.shape[0], ...)])
    for i, chunk in enumerate(a_scale_chunks)
]
a_scale_final = torch.cat(a_scale_padded)
```

### Visual Example of a_scale Padding

```
Original a_scale (2 groups, m=128 each):
┌──────────────────────────────────────────┐
│ Group 0 scales: 128 rows                 │
├──────────────────────────────────────────┤
│ Group 1 scales: 128 rows                 │
└──────────────────────────────────────────┘
Total: 256 rows

After padding (alignment_m_sf = 128):
┌──────────────────────────────────────────┐
│ Group 0 scales: 128 rows (no padding)    │  m_sf[0] = 128
├──────────────────────────────────────────┤
│ Group 1 scales: 128 rows                 │
│ Padding: 128 rows of zeros               │  m_sf[1] = 256
└──────────────────────────────────────────┘
Total: 384 rows

m_indptr_padded = [0, 128, 384]
```

---

## Format Conversion Summary

### Complete Pipeline: GPT OSS → FlashInfer

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Format Conversion Pipeline                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  GPT OSS Storage Format                                              │
│  ┌────────────────────────────────────────┐                         │
│  │ blocks: (experts, n, k//2)  uint8      │                         │
│  │ scales: (experts, n, k//32) uint8      │                         │
│  └────────────────────────────────────────┘                         │
│                      │                                               │
│                      ▼                                               │
│  Step 1: Verify shapes and alignment                                 │
│  ┌────────────────────────────────────────┐                         │
│  │ n_padded = ceil(n / 8) * 8             │                         │
│  │ k_padded = ceil(k / 128) * 128         │                         │
│  │ Pad if necessary                       │                         │
│  └────────────────────────────────────────┘                         │
│                      │                                               │
│                      ▼                                               │
│  Step 2: Swizzle scales                                              │
│  ┌────────────────────────────────────────┐                         │
│  │ b_scale_swizzled = swizzle_blockscale( │                         │
│  │     scales, batch_size, n_padded,      │                         │
│  │     k_padded, tile_size=32             │                         │
│  │ )                                      │                         │
│  └────────────────────────────────────────┘                         │
│                      │                                               │
│                      ▼                                               │
│  FlashInfer Format                                                   │
│  ┌────────────────────────────────────────┐                         │
│  │ b: (batch, n_padded, k_padded//2)      │                         │
│  │ b_scale: (batch, n_padded, k_padded//32)│ SWIZZLED               │
│  └────────────────────────────────────────┘                         │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Activation Quantization (Runtime)

```
┌─────────────────────────────────────────────────────────────────────┐
│                   Activation Quantization                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Input: activations (bfloat16)                                       │
│  ┌────────────────────────────────────────┐                         │
│  │ Shape: (cum_m, k)                      │                         │
│  └────────────────────────────────────────┘                         │
│                      │                                               │
│                      ▼                                               │
│  Step 1: Quantize to MXFP8                                           │
│  ┌────────────────────────────────────────┐                         │
│  │ a_fp8, a_scale = quantize_tensor_mxfp( │                         │
│  │     activations, tile_size=32,         │                         │
│  │     k_padded, is_fp4=False             │                         │
│  │ )                                      │                         │
│  │                                        │                         │
│  │ a_fp8: (cum_m, k_padded) float8        │                         │
│  │ a_scale: (cum_m, k_padded//32) uint8   │                         │
│  └────────────────────────────────────────┘                         │
│                      │                                               │
│                      ▼                                               │
│  Step 2: Swizzle a_scale                                             │
│  ┌────────────────────────────────────────┐                         │
│  │ a_scale_swizzled = swizzle_blockscale( │                         │
│  │     a_scale.unflatten(0, (groups, m)), │                         │
│  │     groups, m, k_padded, tile_size     │                         │
│  │ ).flatten(0, 1)                        │                         │
│  └────────────────────────────────────────┘                         │
│                      │                                               │
│                      ▼                                               │
│  Step 3: Pad a_scale for group GEMM                                  │
│  ┌────────────────────────────────────────┐                         │
│  │ Apply per-group padding to multiples   │                         │
│  │ of 128 rows                            │                         │
│  └────────────────────────────────────────┘                         │
│                      │                                               │
│                      ▼                                               │
│  FlashInfer Format                                                   │
│  ┌────────────────────────────────────────┐                         │
│  │ a: (cum_m, k_padded) float8            │                         │
│  │ a_scale: (cum_m_padded, k_padded//32)  │ SWIZZLED + PADDED       │
│  └────────────────────────────────────────┘                         │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Shape Reference Table

| Tensor | GPT OSS Shape | FlashInfer Shape | Notes |
|--------|---------------|------------------|-------|
| Activations (a) | `(tokens, hidden)` bf16 | `(cum_m, k_padded)` fp8 | Quantized at runtime |
| Activation scales | N/A | `(cum_m_padded_total, k_groups)` uint8 | Swizzled + per-group padded |
| Weights (b) | `(experts, out, in//2)` uint8 | `(batch, n_padded, k_padded//2)` uint8 | Same packing |
| Weight scales | `(experts, out, in//32)` uint8 | `(batch, n_padded, k_groups)` uint8 | Swizzled |
| m_indptr | N/A | `(batch+1,)` int32 | Cumulative row counts |

---

## Example: Complete MXFP4 GEMM

```python
import torch
from flashinfer.gemm import group_gemm_mxfp4_nt_groupwise

# Parameters
m = 128          # Rows per group
n = 256          # Output features  
k = 256          # Input features
group_size = 2   # Number of expert groups
tile_size = 32   # Quantization tile size

# Alignment
alignment_m_sf = 128
n_padded = ((n + 7) // 8) * 8      # = 256
k_padded = ((k + 127) // 128) * 128  # = 256

# 1. Quantize activations to MXFP8
a_fp8, a_scale = quantize_tensor_mxfp(activations, tile_size, None, k_padded, is_fp4=False)

# 2. Quantize weights to MXFP4 (typically done at load time)
b_fp4, b_scale = quantize_tensor_mxfp(weights, tile_size, n_padded, k_padded, is_fp4=True)

# 3. Swizzle both scales
a_scale_swizzled = swizzle_blockscale(
    a_scale.unflatten(0, (group_size, m)), 
    group_size, m, k_padded, tile_size
).flatten(0, 1)

b_scale_swizzled = swizzle_blockscale(
    b_scale, group_size, n_padded, k_padded, tile_size
)

# 4. Create m_indptr
group_arange = torch.arange(0, group_size + 1, dtype=torch.int32)
m_indptr = group_arange * m  # [0, 128, 256]

# 5. Pad a_scale for group GEMM (CRITICAL!)
m_indptr_padded = (
    (m_indptr + group_arange * (alignment_m_sf - 1)) 
    // alignment_m_sf 
    * alignment_m_sf
)
m_sf = m_indptr_padded[1:] - m_indptr_padded[:-1]

a_scale_chunks = a_scale_swizzled.chunk(group_size, dim=0)
a_scale_final = torch.cat([
    torch.cat([chunk, torch.zeros(m_sf[i] - chunk.shape[0], chunk.shape[1])])
    for i, chunk in enumerate(a_scale_chunks)
])

# 6. Run FlashInfer GEMM
output = group_gemm_mxfp4_nt_groupwise(
    a=a_fp8,
    b=b_fp4,
    a_scale=a_scale_final,
    b_scale=b_scale_swizzled,
    m_indptr=m_indptr,
    swap_ab=True,
    out_dtype=torch.bfloat16
)[:, :n]  # Trim to actual n
```

---

## Understanding `swap_ab`

The `swap_ab` parameter in `group_gemm_mxfp4_nt_groupwise` is a **kernel optimization hint**, not a mathematical operation modifier.

### Key Insight: Same Output, Different Execution

Looking at [FlashInfer's official test](https://github.com/flashinfer-ai/flashinfer/blob/main/tests/gemm/test_groupwise_scaled_gemm_mxfp4.py):

```python
swap_ab_list = [True, False]
for mma_sm, tile_m, tile_n, tile_k, swap_ab in product(...):
    out = group_gemm_mxfp4_nt_groupwise(
        a_fp8, b_fp4,
        a_scale_swizzled, b_scale_swizzled,
        m_indptr,
        swap_ab=swap_ab,  # Both True and False tested!
        ...
    )[:, :n]
    torch.testing.assert_close(out, out_ref, atol=1e-2, rtol=1e-2)  # Same result!
```

**Both `swap_ab=True` and `swap_ab=False` produce the same output!**

### What `swap_ab` Actually Does

`swap_ab` controls the **internal kernel execution strategy**, not the mathematical result:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    What swap_ab Controls                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Mathematical operation is ALWAYS:                                  │
│      Output = A @ B.T                                               │
│      Where A is FP8 activations, B is FP4 weights                   │
│                                                                     │
│  swap_ab affects HOW the kernel computes this:                      │
│                                                                     │
│  swap_ab=False:                                                     │
│      - Kernel treats A as "primary" matrix                          │
│      - Tiling/threading optimized for A's access pattern            │
│      - May be faster when M >> N (many tokens, few features)        │
│                                                                     │
│  swap_ab=True:                                                      │
│      - Kernel treats B as "primary" matrix                          │
│      - Tiling/threading optimized for B's access pattern            │
│      - May be faster when N >> M (few tokens, many features)        │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Why It Exists: Performance Optimization

Matrix multiplication kernels can be "transposed" internally without changing the result:

```
C = A @ B.T

Can be computed as:
  1. Direct: iterate over A's rows, accumulate with B's rows
  2. Swapped: iterate over B's rows, accumulate with A's rows (then transpose output layout)

Both give the same C, but with different memory access patterns.
```

### When to Use Which

| Scenario | Recommended | Reason |
|----------|-------------|--------|
| M is large (many tokens) | Try both, benchmark | Access pattern for A matters more |
| N is large (many features) | Try both, benchmark | Access pattern for B matters more |
| Unsure | Either works | Same correctness, may differ in speed |

### For MoE Implementation

Since both produce correct results, you can:

1. **Start with either** (we use `swap_ab=True` in our sanity check)
2. **Benchmark both** on your actual workload
3. **Choose the faster one** for production

```python
# Both are correct!
output = group_gemm_mxfp4_nt_groupwise(
    a_fp8, b_fp4,
    a_scale_swizzled, b_scale_swizzled,
    m_indptr,
    swap_ab=True,  # or False - same result, possibly different speed
    out_dtype=torch.bfloat16
)
```

### Summary

| Aspect | `swap_ab=False` | `swap_ab=True` |
|--------|-----------------|----------------|
| **Output** | Same | Same |
| **Correctness** | ✓ | ✓ |
| **Speed** | Depends on shapes | Depends on shapes |
| **Use case** | Performance tuning | Performance tuning |

**Bottom line**: `swap_ab` is a performance knob, not a correctness knob. Both values produce mathematically identical results.

---

## Column-Major vs Row-Major Scale Layouts

The FlashInfer documentation mentions that `a_scale` uses "column-major" layout while `b_scale` uses "row-major" layout. Let's understand what this means.

### Memory Layout Basics

For a 2D tensor with shape `(rows, cols)`:

```
Row-Major (C-style, PyTorch default):
┌─────────────────────────────────────────┐
│ Elements of each ROW are contiguous     │
│                                         │
│ Logical view:     Memory layout:        │
│ ┌───┬───┬───┐     [a, b, c, d, e, f]    │
│ │ a │ b │ c │      ↑─row0─↑ ↑─row1─↑    │
│ ├───┼───┼───┤                           │
│ │ d │ e │ f │     Stride: (cols, 1)     │
│ └───┴───┴───┘     = (3, 1) for 2×3      │
└─────────────────────────────────────────┘

Column-Major (Fortran-style):
┌─────────────────────────────────────────┐
│ Elements of each COLUMN are contiguous  │
│                                         │
│ Logical view:     Memory layout:        │
│ ┌───┬───┬───┐     [a, d, b, e, c, f]    │
│ │ a │ b │ c │      ↑col0↑ ↑col1↑ ↑col2↑ │
│ ├───┼───┼───┤                           │
│ │ d │ e │ f │     Stride: (1, rows)     │
│ └───┴───┴───┘     = (1, 2) for 2×3      │
└─────────────────────────────────────────┘
```

### Why Different Layouts?

The GPU kernel accesses `a_scale` and `b_scale` differently during the GEMM computation:

```
GEMM: C = A @ B.T

For each output element C[m, n]:
    C[m, n] = Σ_k (A[m, k] × A_scale[m, k//32]) × (B[n, k] × B_scale[n, k//32])
```

The kernel is optimized with different access patterns:

#### For `a_scale` (Activations): Column-Major

```
During computation, multiple threads process DIFFERENT ROWS of A
but the SAME K-block simultaneously:

Thread 0: processes A[row=0, k_block=5] → needs a_scale[0, 5]
Thread 1: processes A[row=1, k_block=5] → needs a_scale[1, 5]
Thread 2: processes A[row=2, k_block=5] → needs a_scale[2, 5]
...

Column-major layout puts [0,5], [1,5], [2,5], ... contiguous in memory!
This enables coalesced memory access (all threads read adjacent addresses).

┌─────────────────────────────────────────────────────────────────────┐
│  a_scale column-major: (cum_m, k_groups)                            │
│                                                                     │
│  Memory: [s(0,0), s(1,0), s(2,0), ..., s(0,1), s(1,1), s(2,1), ...] │
│           ↑──── column 0 ────↑      ↑──── column 1 ────↑            │
│                                                                     │
│  When threads read k_group=5, they access:                          │
│  s(0,5), s(1,5), s(2,5), ... → CONTIGUOUS! → Fast coalesced read   │
└─────────────────────────────────────────────────────────────────────┘
```

#### For `b_scale` (Weights): Row-Major

```
During computation, threads process the SAME ROW of B (same output feature)
across DIFFERENT K-blocks:

For output feature n=7:
  Need: B[7, k=0..31]   × b_scale[7, 0]
        B[7, k=32..63]  × b_scale[7, 1]
        B[7, k=64..95]  × b_scale[7, 2]
        ...

Row-major layout puts [7,0], [7,1], [7,2], ... contiguous in memory!

┌─────────────────────────────────────────────────────────────────────┐
│  b_scale row-major: (batch, n, k_groups)                            │
│                                                                     │
│  Memory: [s(0,0), s(0,1), s(0,2), ..., s(1,0), s(1,1), s(1,2), ...] │
│           ↑────── row 0 ──────↑      ↑────── row 1 ──────↑          │
│                                                                     │
│  When processing output row 7, access:                              │
│  s(7,0), s(7,1), s(7,2), ... → CONTIGUOUS! → Fast sequential read  │
└─────────────────────────────────────────────────────────────────────┘
```

### In Practice: Swizzling Handles It

**You don't need to manually create column-major tensors!**

The `swizzle_blockscale` function transforms the scales into the correct memory layout:

```python
# You create scales in normal PyTorch row-major format
a_scale = ...  # Shape: (cum_m, k_groups), row-major (default)
b_scale = ...  # Shape: (batch, n, k_groups), row-major (default)

# Swizzling transforms to the layout the kernel expects
a_scale_swizzled = swizzle_blockscale(a_scale.unflatten(...), ...)  # → column-major + interleaved
b_scale_swizzled = swizzle_blockscale(b_scale, ...)                 # → row-major + interleaved
```

The swizzle operation:
1. Pads dimensions for alignment
2. Interleaves data for optimal GPU cache line usage
3. Implicitly handles the column-major/row-major transformation

### Visual Summary

```
┌─────────────────────────────────────────────────────────────────────┐
│                     Scale Layout Summary                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  a_scale (activations):                                             │
│    Logical shape: (cum_m, k_groups)                                 │
│    Memory layout: Column-major (after swizzle)                      │
│    Access pattern: Different rows, same k_group → coalesced         │
│                                                                     │
│  b_scale (weights):                                                 │
│    Logical shape: (batch, n, k_groups)                              │
│    Memory layout: Row-major (after swizzle)                         │
│    Access pattern: Same row, different k_groups → sequential        │
│                                                                     │
│  Key insight: swizzle_blockscale() handles the transformation!      │
│  You just provide normal PyTorch tensors, it does the rest.         │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### PyTorch Strides Reference

For a tensor with shape `(M, K)`:

| Layout | Stride | Memory Order | PyTorch |
|--------|--------|--------------|---------|
| Row-major | `(K, 1)` | Row by row | Default |
| Column-major | `(1, M)` | Column by column | `.T.contiguous().T` or `.t().contiguous().t()` |

Example:
```python
# Row-major (default)
t = torch.tensor([[1, 2, 3], [4, 5, 6]])  # Shape (2, 3)
# Memory: [1, 2, 3, 4, 5, 6]
# Stride: (3, 1)

# Column-major
t_colmajor = t.T.contiguous().T  # Same shape (2, 3), different memory layout
# Memory: [1, 4, 2, 5, 3, 6]  
# Stride: (1, 2)
```

---

## Runtime Activation Quantization (bfloat16 → MXFP8)

At runtime, activations arrive as bfloat16 and must be quantized to MXFP8 before calling `group_gemm_mxfp4_nt_groupwise`.

### FlashInfer's Native Solution: `mxfp8_quantize`

FlashInfer provides a highly optimized CUDA kernel for this:

```python
from flashinfer.fp8_quantization import mxfp8_quantize

# Input: bfloat16 activations
activations = torch.randn(1024, 2048, dtype=torch.bfloat16, device='cuda')

# Quantize to MXFP8
a_fp8, a_scale = mxfp8_quantize(
    activations,
    is_sf_swizzled_layout=False,  # Get non-swizzled scales for manual swizzling
    alignment=32,                  # tile_size
)
```

### Function Signature

```python
mxfp8_quantize(
    input: torch.Tensor,           # Shape: (M, K), dtype: fp16/bf16
    is_sf_swizzled_layout: bool,   # Whether to pre-swizzle scales
    alignment: int = 32,           # tile_size (scale factor vector size)
) -> Tuple[torch.Tensor, torch.Tensor]:
    # Returns:
    #   - Quantized tensor: (M, K), dtype: float8_e4m3fn
    #   - Scale factors: flattened uint8 tensor
```

### Performance

Benchmarked on NVIDIA Blackwell GPU:

```
Input: (1024, 2048) bfloat16
Output: (1024, 2048) float8_e4m3fn + scales
Time: ~0.016 ms per call
Throughput: ~263 GB/s
```

This is **extremely fast** — the quantization overhead is negligible compared to the GEMM computation.

### Scale Layout Options

| `is_sf_swizzled_layout` | Scale Shape | Use Case |
|-------------------------|-------------|----------|
| `True` | 1D swizzled | Direct use with single-batch GEMM |
| `False` | 1D flat (reshapable to M×k_groups) | Multi-group GEMM with manual swizzling |

### Recommended Approach for Group GEMM

For `group_gemm_mxfp4_nt_groupwise` with multiple groups:

```python
from flashinfer.fp8_quantization import mxfp8_quantize

# 1. Quantize activations (non-swizzled for manual control)
a_fp8, a_scale_flat = mxfp8_quantize(
    activations,  # (cum_m, K) bfloat16
    is_sf_swizzled_layout=False
)

# 2. Reshape scale to (cum_m, k_groups)
k_groups = K // 32
a_scale = a_scale_flat.reshape(cum_m, k_groups)

# 3. Reshape for swizzling: (group_size, m_per_group, k_groups)
a_scale_3d = a_scale.unflatten(0, (group_size, m_per_group))

# 4. Apply swizzle + padding
a_scale_swizzled = swizzle_blockscale(a_scale_3d, group_size, m_per_group, k_padded, tile_size)
a_scale_swizzled = a_scale_swizzled.flatten(0, 1)

# 5. Apply per-group padding (see Group GEMM Requirements section)
# ... padding logic ...

# 6. Run GEMM
output = group_gemm_mxfp4_nt_groupwise(a_fp8, b_fp4, a_scale_final, b_scale_swizzled, m_indptr, ...)
```

### Alternative: Use Swizzled Layout Directly (Single Group)

For simpler cases with a single group:

```python
# Swizzled layout - scale is already in correct format
a_fp8, a_scale_swizzled = mxfp8_quantize(
    activations,
    is_sf_swizzled_layout=True  # Pre-swizzled!
)

# May still need padding for alignment
```

### Comparison: Native vs Manual Quantization

| Approach | Speed | Complexity | Recommended |
|----------|-------|------------|-------------|
| `mxfp8_quantize` (FlashInfer) | ~0.016ms | Low | ✅ **Yes** |
| Manual PyTorch quantization | ~0.5-1ms | High | No |
| Custom CUDA kernel | Variable | Very High | Only if needed |

**Always use `mxfp8_quantize`** — it's optimized, handles edge cases, and produces exactly the format FlashInfer expects.

---

## Key Takeaways

1. **MXFP4 = FP4 E2M1 values + UE8M0 block scales** (32 values per scale)

2. **Nibble packing**: Even indices in high nibble, odd indices in low nibble

3. **Both scales must be swizzled** using `block_scale_interleave_sm100`

4. **a_scale requires per-group padding** to multiples of 128 rows

5. **Alignment requirements**: M÷4, N÷8, K÷128

6. **`swap_ab` is a performance hint**, not a correctness flag — both `True` and `False` produce identical results

7. **Column-major vs row-major**: Different layouts for `a_scale` and `b_scale` optimize GPU memory access patterns — but `swizzle_blockscale()` handles this automatically

8. **Use `mxfp8_quantize` for runtime activation quantization** — it's fast (~0.016ms), native, and produces the correct format

