
import torch
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from flashinfer_metal.attention import AttentionCompiler
from flashinfer_metal.utils import validate_mps_device

def test_bf16_attention():
    print("Initializing AttentionCompiler...")
    compiler = AttentionCompiler(page_size=16)
    
    if not compiler.can_use_mps_kernels():
        print("Skipping verification: MPS kernels not available.")
        return

    print("Generating inputs...")
    torch.manual_seed(42)
    
    # Dimensions
    num_tokens = 32
    num_heads = 4
    head_dim = 64
    num_kv_heads = 4
    page_size = 16
    
    # Create inputs
    query = torch.randn(num_tokens, num_heads, head_dim, dtype=torch.bfloat16, device="mps")
    
    # KV Cache setup
    num_pages = 10
    kv_cache = torch.randn(num_pages, 2, page_size, num_kv_heads, head_dim, dtype=torch.bfloat16, device="mps")
    
    # Metadata
    kv_page_indices = torch.arange(num_pages, dtype=torch.int32, device="mps")
    kv_page_indptr = torch.tensor([0, num_pages], dtype=torch.int32, device="mps")
    kv_last_page_lens = torch.tensor([page_size], dtype=torch.int32, device="mps")
    qo_indptr = torch.tensor([0, num_tokens], dtype=torch.int32, device="mps")
    
    print("Running bfloat16 attention...")
    output_bf16 = compiler.run_attention_mps(
        query, kv_cache, kv_page_indices, kv_page_indptr, kv_last_page_lens, qo_indptr
    )
    
    print("Running float32 reference attention...")
    query_f32 = query.float()
    kv_cache_f32 = kv_cache.float()
    
    output_f32 = compiler.run_attention_mps(
        query_f32, kv_cache_f32, kv_page_indices, kv_page_indptr, kv_last_page_lens, qo_indptr
    )
    
    print("Comparing results...")
    # Compare bf16 result with f32 result (converted to bf16 for fair comparison or checking closeness)
    diff = (output_bf16.float() - output_f32).abs().max()
    print(f"Max difference: {diff}")
    
    # BF16 has lower precision, so tolerance is looser.
    if diff < 1e-2:
        print("✅ Verification PASSED: bfloat16 output matches float32 reference.")
    else:
        print("❌ Verification FAILED: Difference too large.")
        sys.exit(1)

if __name__ == "__main__":
    try:
        test_bf16_attention()
    except Exception as e:
        print(f"❌ Error during verification: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
