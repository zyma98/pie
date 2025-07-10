#include "l4ma.cuh"
#include "config.hpp"
#include "common.cuh"   // Your helper functions header

#include <stdexcept>
#include <iostream>
#include <utility>
#include <algorithm> // for std::max

#include "flashinfer/norm.cuh"
#include "flashinfer/pos_enc.cuh"
#include "flashinfer/page.cuh"
#include "flashinfer/activation.cuh"
#include "flashinfer_ops.cuh"

// --- Helper CUDA Kernels ---
template <typename T>
__global__ void add_residual_kernel(T* x, const T* residual, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        x[idx] = x[idx] + residual[idx];
    }
}

__device__ __forceinline__ float silu(const float &val) { return val / (1.0f + __expf(-val)); }

template <typename T>
void silu_and_mul(
    T *out_ptr,
    const T *in_ptr,
    int num_tokens,
    int d_half,
    cudaStream_t stream)
{
    uint32_t vec_size = 16 / sizeof(T);
    cudaLaunchConfig_t config;
    config.gridDim = num_tokens;
    config.blockDim = std::min(d_half / vec_size, 1024U);
    config.dynamicSmemBytes = 0;
    config.stream = stream;
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    config.numAttrs = 1;
    config.attrs = attrs;

    auto kernel = flashinfer::activation::act_and_mul_kernel<T, silu>;
    cudaLaunchKernelEx(&config, kernel, out_ptr, in_ptr, d_half);
}

// --- Constructor Implementations ---
template <typename T>
RMSNorm<T>::RMSNorm(const L4maConfig& config)
    : config_(config), weight_(std::make_shared<thrust::device_vector<T>>(config.hidden_size)) {}

template <typename T>
L4maMlp<T>::L4maMlp(const L4maConfig& config)
    : config_(config),
      gate_proj_weights_(std::make_shared<thrust::device_vector<T>>(config.hidden_size * config.intermediate_size)),
      up_proj_weights_(std::make_shared<thrust::device_vector<T>>(config.hidden_size * config.intermediate_size)),
      down_proj_weights_(std::make_shared<thrust::device_vector<T>>(config.intermediate_size * config.hidden_size)) {}

template <typename T>
L4maAttention<T>::L4maAttention(const L4maConfig& config)
    : config_(config),
      q_proj_weights_(std::make_shared<thrust::device_vector<T>>(config.hidden_size * (config.num_query_heads * config.head_size))),
      k_proj_weights_(std::make_shared<thrust::device_vector<T>>(config.hidden_size * (config.num_key_value_heads * config.head_size))),
      v_proj_weights_(std::make_shared<thrust::device_vector<T>>(config.hidden_size * (config.num_key_value_heads * config.head_size))),
      o_proj_weights_(std::make_shared<thrust::device_vector<T>>((config.num_query_heads * config.head_size) * config.hidden_size)) {
    if (config_.use_qkv_bias) {
        q_proj_bias_ = std::make_shared<thrust::device_vector<T>>(config.num_query_heads * config.head_size);
        k_proj_bias_ = std::make_shared<thrust::device_vector<T>>(config.num_key_value_heads * config.head_size);
        v_proj_bias_ = std::make_shared<thrust::device_vector<T>>(config.num_key_value_heads * config.head_size);
    }
}

template <typename T>
L4maDecoderLayer<T>::L4maDecoderLayer(const L4maConfig& config)
    : config_(config),
      self_attn_(config),
      mlp_(config),
      input_layernorm_(config),
      post_attention_layernorm_(config) {}

template <typename T>
L4maModel<T>::L4maModel(const L4maConfig& config)
    : config_(config),
      embed_tokens_weight_(std::make_shared<thrust::device_vector<T>>(config.vocab_size * config.hidden_size)),
      norm_(config) {

    layers_.reserve(config.num_layers);
    for (int i = 0; i < config.num_layers; ++i) {
        layers_.emplace_back(config);
    }
}

template <typename T>
L4maForCausalLM<T>::L4maForCausalLM(const L4maConfig& config)
    : config_(config), model_(config) {
    CUBLAS_CHECK(cublasLtCreate(&cublaslt_handle_));
}

// --- KV Cache and Workspace Management ---

template <typename T>
void L4maForCausalLM<T>::create_kv_device_vectors(int max_kv_num) {
    size_t kv_cache_size = static_cast<size_t>(max_kv_num) * config_.num_key_value_heads * config_.head_size;
    if (kv_cache_k_.size() != kv_cache_size) {
        kv_cache_k_.resize(kv_cache_size);
    }
    if (kv_cache_v_.size() != kv_cache_size) {
        kv_cache_v_.resize(kv_cache_size);
    }
}

template <typename T>
size_t L4maForCausalLM<T>::get_workspace_size(int max_num_tokens) const {
    size_t hidden_states_size = (size_t)max_num_tokens * config_.hidden_size;
    size_t residual_size = (size_t)max_num_tokens * config_.hidden_size;
    size_t normed_size = (size_t)max_num_tokens * config_.hidden_size;
    size_t output_size = (size_t)max_num_tokens * config_.hidden_size; 
    size_t attn_temp_size = (size_t)max_num_tokens * (config_.num_query_heads * config_.head_size + 2 * config_.num_key_value_heads * config_.head_size);
    size_t mlp_temp_size = (size_t)max_num_tokens * 2 * config_.intermediate_size;
    size_t temp_size = std::max(attn_temp_size, mlp_temp_size);
    
    size_t total_elements = hidden_states_size + residual_size + normed_size + output_size + temp_size;
    size_t model_workspace_bytes = total_elements * sizeof(T);
    size_t flashinfer_workspace_bytes = 256 * 1024 * 1024;

    return std::max(model_workspace_bytes, flashinfer_workspace_bytes);
}


// --- get_parameters() Implementations ---

template <typename T>
std::map<std::string, std::shared_ptr<thrust::device_vector<T>>> RMSNorm<T>::get_parameters() {
    return {{"weight", weight_}};
}

template <typename T>
std::map<std::string, std::shared_ptr<thrust::device_vector<T>>> L4maMlp<T>::get_parameters() {
    return {{"gate_proj.weight", gate_proj_weights_},
            {"up_proj.weight", up_proj_weights_},
            {"down_proj.weight", down_proj_weights_}};
}

template <typename T>
std::map<std::string, std::shared_ptr<thrust::device_vector<T>>> L4maAttention<T>::get_parameters() {
    auto params = std::map<std::string, std::shared_ptr<thrust::device_vector<T>>>{
        {"q_proj.weight", q_proj_weights_},
        {"k_proj.weight", k_proj_weights_},
        {"v_proj.weight", v_proj_weights_},
        {"o_proj.weight", o_proj_weights_}};
    if (config_.use_qkv_bias) {
        params["q_proj.bias"] = q_proj_bias_;
        params["k_proj.bias"] = k_proj_bias_;
        params["v_proj.bias"] = v_proj_bias_;
    }
    return params;
}

template <typename T>
std::map<std::string, std::shared_ptr<thrust::device_vector<T>>> L4maDecoderLayer<T>::get_parameters() {
    std::map<std::string, std::shared_ptr<thrust::device_vector<T>>> params;
    for (auto const& [key, val] : self_attn_.get_parameters()) { params["self_attn." + key] = val; }
    for (auto const& [key, val] : mlp_.get_parameters()) { params["mlp." + key] = val; }
    for (auto const& [key, val] : input_layernorm_.get_parameters()) { params["input_layernorm." + key] = val; }
    for (auto const& [key, val] : post_attention_layernorm_.get_parameters()) { params["post_attention_layernorm." + key] = val; }
    return params;
}

template <typename T>
std::map<std::string, std::shared_ptr<thrust::device_vector<T>>>  L4maModel<T>::get_parameters() {
    std::map<std::string, std::shared_ptr<thrust::device_vector<T>>> params;
    params["embed_tokens.weight"] = embed_tokens_weight_;
    for (size_t i = 0; i < layers_.size(); ++i) {
        for (auto const& [key, val] : layers_[i].get_parameters()) {
            params["layers." + std::to_string(i) + "." + key] = val;
        }
    }
    for (auto const& [key, val] : norm_.get_parameters()) { params["norm." + key] = val; }
    return params;
}

template <typename T>
std::map<std::string, std::shared_ptr<thrust::device_vector<T>>> L4maForCausalLM<T>::get_parameters() {
    std::map<std::string, std::shared_ptr<thrust::device_vector<T>>> params;
    for (auto const& [key, val] : model_.get_parameters()) {
        params["model." + key] = val;
    }
    params["lm_head.weight"] = lm_head_weight_;
    return params;
}

// --- Forward Pass Implementations ---

template <typename T>
void RMSNorm<T>::forward(
    T* output,
    const T* input,
    int num_tokens,
    cudaStream_t stream) {
    
    uint32_t d = config_.hidden_size;

    flashinfer::norm::RMSNorm(
        const_cast<T *>(input),
        const_cast<T *>(thrust::raw_pointer_cast(weight_->data())),
        output,
        num_tokens, d, d, d, config_.rms_norm_eps, false, stream
    );
}

template <typename T>
void L4maMlp<T>::forward(
    T* output, 
    const T* x, 
    int num_tokens, 
    T* temp_buffer,
    cublasLtHandle_t ltHandle, 
    cudaStream_t stream, 
    char* workspace_buffer_float
) {
    const int hidden_size = config_.hidden_size;
    const int intermediate_size = config_.intermediate_size;
    const size_t workspace_size = (size_t)num_tokens * config_.hidden_size * 4; // A safe upper bound

    T* gate_proj_out_ptr = temp_buffer;
    T* up_proj_out_ptr = temp_buffer + (size_t)num_tokens * intermediate_size;

    // 1. Gate projection
    gemm_cublasLt<T>(ltHandle, stream, x, thrust::raw_pointer_cast(gate_proj_weights_->data()), nullptr, gate_proj_out_ptr, num_tokens, intermediate_size, hidden_size, workspace_buffer_float, workspace_size, false, true);
    
    // 2. Up projection
    gemm_cublasLt<T>(ltHandle, stream, x, thrust::raw_pointer_cast(up_proj_weights_->data()), nullptr, up_proj_out_ptr, num_tokens, intermediate_size, hidden_size, workspace_buffer_float, workspace_size, false, true);

    // 3. SwiGLU activation
    silu_and_mul<T>(
        gate_proj_out_ptr,
        temp_buffer,
        num_tokens,
        intermediate_size,
        stream
    );

    // 4. Down projection
    gemm_cublasLt<T>(ltHandle, stream, gate_proj_out_ptr, thrust::raw_pointer_cast(down_proj_weights_->data()), nullptr, output, num_tokens, hidden_size, intermediate_size, workspace_buffer_float, workspace_size, false, true);
}

template <typename T>
void L4maAttention<T>::forward(
    T* attn_output,
    const T* hidden_states,
    const thrust::device_vector<int32_t>& position_ids,
    thrust::device_vector<T>& kv_cache_k,
    thrust::device_vector<T>& kv_cache_v,
    thrust::device_vector<int32_t>& kv_page_indices,
    thrust::device_vector<int32_t>& kv_page_indptr,
    thrust::device_vector<int32_t>& kv_last_page_lens,
    thrust::device_vector<int32_t>& qo_indptr,
    thrust::device_vector<uint8_t>& custom_mask,
    thrust::device_vector<int32_t>& mask_indptr,
    T* temp_buffer,
    cublasLtHandle_t ltHandle,
    cudaStream_t stream,
    char* workspace_buffer_float,
    flashinfer::BatchPrefillHandler& prefill_handler,
    const int32_t page_size,
    thrust::device_vector<int32_t>& kv_batch_indices,
    thrust::device_vector<int32_t>& kv_positions
) {

    const int batch_size = (position_ids.size() > 0) ? (qo_indptr.size() - 1) : 0;
    const int num_tokens = position_ids.size();
    const int hidden_size = config_.hidden_size;
    const int head_size = config_.head_size;
    const int num_query_heads = config_.num_query_heads;
    const int num_key_value_heads = config_.num_key_value_heads;
    const size_t workspace_size = (size_t)num_tokens * config_.hidden_size * 4; // A safe upper bound
    
    size_t q_size = (size_t)num_tokens * num_query_heads * head_size;
    size_t k_size = (size_t)num_tokens * num_key_value_heads * head_size;
    
    T* q_proj_ptr = temp_buffer;
    T* k_proj_ptr = q_proj_ptr + q_size;
    T* v_proj_ptr = k_proj_ptr + k_size;
    
    // 1. Q, K, V projections
    gemm_cublasLt<T>(ltHandle, stream, hidden_states, thrust::raw_pointer_cast(q_proj_weights_->data()), config_.use_qkv_bias ? thrust::raw_pointer_cast(q_proj_bias_->data()) : nullptr, q_proj_ptr, num_tokens, num_query_heads * head_size, hidden_size, workspace_buffer_float, workspace_size, false, true);
    gemm_cublasLt<T>(ltHandle, stream, hidden_states, thrust::raw_pointer_cast(k_proj_weights_->data()), config_.use_qkv_bias ? thrust::raw_pointer_cast(k_proj_bias_->data()) : nullptr, k_proj_ptr, num_tokens, num_key_value_heads * head_size, hidden_size, workspace_buffer_float, workspace_size, false, true);
    gemm_cublasLt<T>(ltHandle, stream, hidden_states, thrust::raw_pointer_cast(v_proj_weights_->data()), config_.use_qkv_bias ? thrust::raw_pointer_cast(v_proj_bias_->data()) : nullptr, v_proj_ptr, num_tokens, num_key_value_heads * head_size, hidden_size, workspace_buffer_float, workspace_size, false, true);

    // 2. Apply RoPE (in-place)
    flashinfer::BatchQKApplyLlama31RotaryPosIds(
        q_proj_ptr, k_proj_ptr, q_proj_ptr, k_proj_ptr,
        thrust::raw_pointer_cast(position_ids.data()),
        num_tokens, num_query_heads, num_key_value_heads, head_size, head_size,
        num_query_heads * head_size, head_size, num_key_value_heads * head_size, head_size,
        num_query_heads * head_size, head_size, num_key_value_heads * head_size, head_size,
        false, config_.rope_factor, config_.rope_theta, config_.rope_low_frequency_factor,
        config_.rope_high_frequency_factor, 8192, stream
    );

    // 3. Paged KV-cache operations
    flashinfer::paged_kv_t<T, int32_t> paged_kv(
        num_key_value_heads, page_size, head_size, batch_size,
        flashinfer::QKVLayout::kNHD,
        thrust::raw_pointer_cast(kv_cache_k.data()),
        thrust::raw_pointer_cast(kv_cache_v.data()),
        thrust::raw_pointer_cast(kv_page_indices.data()), 
        thrust::raw_pointer_cast(kv_page_indptr.data()), 
        thrust::raw_pointer_cast(kv_last_page_lens.data())
    );

    flashinfer::AppendPagedKVCache<T, int32_t>(
        paged_kv, k_proj_ptr, v_proj_ptr,
        thrust::raw_pointer_cast(kv_batch_indices.data()),
        thrust::raw_pointer_cast(kv_positions.data()),
        kv_batch_indices.size(),
        num_key_value_heads * head_size, head_size,
        num_key_value_heads * head_size, head_size);

    T* o_proj_ptr = q_proj_ptr; // Reuse buffer
    flashinfer::BatchPrefillWithPagedKVCacheWrapper<T, T, T, int32_t>(
        &prefill_handler, o_proj_ptr, thrust::raw_pointer_cast(qo_indptr.data()),
        nullptr, paged_kv, o_proj_ptr, nullptr, num_query_heads,
        flashinfer::MaskMode::kCustom,
        thrust::raw_pointer_cast(custom_mask.data()),
        thrust::raw_pointer_cast(mask_indptr.data()),
        flashinfer::PosEncodingMode::kNone);

    gemm_cublasLt<T>(ltHandle, stream, o_proj_ptr, thrust::raw_pointer_cast(o_proj_weights_->data()), nullptr, attn_output, num_tokens, hidden_size, num_query_heads * head_size, workspace_buffer_float, workspace_size, false, true);
}

template <typename T>
void L4maDecoderLayer<T>::forward(
    T* hidden_states,
    const thrust::device_vector<int32_t>& position_ids,
    thrust::device_vector<T>& kv_cache_k,
    thrust::device_vector<T>& kv_cache_v,
    thrust::device_vector<int32_t>& kv_page_indices,
    thrust::device_vector<int32_t>& kv_page_indptr,
    thrust::device_vector<int32_t>& kv_last_page_lens,
    thrust::device_vector<int32_t>& qo_indptr,
    thrust::device_vector<uint8_t>& custom_mask,
    thrust::device_vector<int32_t>& mask_indptr,
    cublasLtHandle_t ltHandle,
    cudaStream_t stream,
    char* workspace_buffer_float,
    flashinfer::BatchPrefillHandler& prefill_handler,
    const int32_t page_size,
    thrust::device_vector<int32_t>& kv_batch_indices,
    thrust::device_vector<int32_t>& kv_positions
) {
    const int num_tokens = position_ids.size();
    const size_t hidden_size_num_elements = (size_t)num_tokens * config_.hidden_size;
    const size_t hidden_size_bytes = hidden_size_num_elements * sizeof(T);

    T* layer_scratch_ptr = reinterpret_cast<T*>(workspace_buffer_float + hidden_size_bytes);
    T* residual_ptr = layer_scratch_ptr;
    T* normed_input_ptr = residual_ptr + hidden_size_num_elements;
    T* attn_output_ptr = normed_input_ptr + hidden_size_num_elements;
    T* mlp_output_ptr = attn_output_ptr; 
    T* temp_ptr = attn_output_ptr + hidden_size_num_elements;

    // --- 1. Self-Attention Block ---
    cudaMemcpyAsync(residual_ptr, hidden_states, hidden_size_bytes, cudaMemcpyDeviceToDevice, stream);
    input_layernorm_.forward(normed_input_ptr, hidden_states, num_tokens, stream);
    self_attn_.forward(attn_output_ptr, normed_input_ptr, position_ids, kv_cache_k, kv_cache_v, 
                       kv_page_indices, kv_page_indptr, kv_last_page_lens, qo_indptr, custom_mask, mask_indptr, temp_ptr, 
                       ltHandle, stream, workspace_buffer_float, prefill_handler, page_size,
                       kv_batch_indices, kv_positions);
    add_residual_kernel<<<(hidden_size_num_elements + 255) / 256, 256, 0, stream>>>(
        hidden_states, attn_output_ptr, hidden_size_num_elements);

    // --- 2. MLP Block ---
    cudaMemcpyAsync(residual_ptr, hidden_states, hidden_size_bytes, cudaMemcpyDeviceToDevice, stream);
    post_attention_layernorm_.forward(normed_input_ptr, hidden_states, num_tokens, stream);
    mlp_.forward(mlp_output_ptr, normed_input_ptr, num_tokens, temp_ptr, ltHandle, stream, workspace_buffer_float);
    add_residual_kernel<<<(hidden_size_num_elements + 255) / 256, 256, 0, stream>>>(
        hidden_states, mlp_output_ptr, hidden_size_num_elements);
}

template <typename T>
void L4maModel<T>::forward(
    T* hidden_states,
    const thrust::device_vector<uint32_t>& input_ids,
    const thrust::device_vector<int32_t>& position_ids,
    thrust::device_vector<T>& kv_cache_k,
    thrust::device_vector<T>& kv_cache_v,
    thrust::device_vector<int32_t>& kv_page_indices,
    thrust::device_vector<int32_t>& kv_page_indptr,
    thrust::device_vector<int32_t>& kv_last_page_lens,
    thrust::device_vector<int32_t>& qo_indptr,
    thrust::device_vector<uint8_t>& custom_mask,
    thrust::device_vector<int32_t>& mask_indptr,
    cublasLtHandle_t ltHandle,
    cudaStream_t stream,
    char* workspace_buffer_float,
    flashinfer::BatchPrefillHandler& prefill_handler,
    const int32_t page_size,
    thrust::device_vector<int32_t>& kv_batch_indices,
    thrust::device_vector<int32_t>& kv_positions
) {
    const int num_tokens = input_ids.size();
    
    embed<T>(
        thrust::raw_pointer_cast(embed_tokens_weight_->data()),
        embed_tokens_weight_->size() / config_.hidden_size,
        thrust::raw_pointer_cast(input_ids.data()),
        input_ids.size(),
        hidden_states, 
        config_.hidden_size, 
        stream
    );

    for (auto& layer : layers_) {
        layer.forward(hidden_states, position_ids, kv_cache_k, kv_cache_v,
                      kv_page_indices, kv_page_indptr, kv_last_page_lens,
                      qo_indptr, custom_mask, mask_indptr, ltHandle, stream, workspace_buffer_float,
                      prefill_handler, page_size, kv_batch_indices, kv_positions);
    }

    size_t hidden_states_bytes = ((size_t)num_tokens * config_.hidden_size) * sizeof(T);
    T* final_norm_input_ptr = reinterpret_cast<T*>(workspace_buffer_float + hidden_states_bytes);
    cudaMemcpyAsync(final_norm_input_ptr, hidden_states, hidden_states_bytes, cudaMemcpyDeviceToDevice, stream);
    norm_.forward(hidden_states, final_norm_input_ptr, num_tokens, stream);
}

template <typename T>
void L4maForCausalLM<T>::forward(
    thrust::device_vector<T>& logits, 
    const thrust::device_vector<uint32_t>& input_ids,
    const thrust::device_vector<int32_t>& position_ids,
    thrust::device_vector<int32_t>& kv_page_indices,
    thrust::device_vector<int32_t>& kv_page_indptr,
    std::vector<int32_t>& kv_page_indptr_host,
    thrust::device_vector<int32_t>& kv_last_page_lens,
    thrust::device_vector<int32_t>& qo_indptr,
    std::vector<int32_t>& qo_indptr_host,
    thrust::device_vector<uint8_t>& custom_mask,
    thrust::device_vector<int32_t>& mask_indptr,
    cudaStream_t stream,
    char* workspace_buffer_float,
    char* workspace_buffer_int,
    const int32_t page_size,
    thrust::device_vector<int32_t>& kv_batch_indices,
    thrust::device_vector<int32_t>& kv_positions
    ) {
    
    const int num_tokens = input_ids.size();
    const size_t workspace_size = get_workspace_size(num_tokens);
    flashinfer::BatchPrefillHandler handler;

    handler.Plan<T, int32_t>(
        (void *)workspace_buffer_float, workspace_size,
        (void *)workspace_buffer_int, (size_t)8 * 1024 * 1024,
        qo_indptr_host.data(), 
        kv_page_indptr_host.data(),
        num_tokens,
        qo_indptr_host.size() - 1,
        config_.num_query_heads,
        config_.num_key_value_heads,
        config_.head_size,
        page_size);

    T* hidden_states_ptr = reinterpret_cast<T*>(workspace_buffer_float);

    model_.forward(
        hidden_states_ptr, input_ids, position_ids,
        kv_cache_k_, kv_cache_v_,
        kv_page_indices, kv_page_indptr, kv_last_page_lens,
        qo_indptr, custom_mask, mask_indptr,
        cublaslt_handle_, stream, workspace_buffer_float, handler, page_size,
        kv_batch_indices, kv_positions
    );

    gemm_cublasLt<T>(
        cublaslt_handle_, stream,
        hidden_states_ptr,
        thrust::raw_pointer_cast(lm_head_weight_->data()),
        nullptr,
        thrust::raw_pointer_cast(logits.data()),
        num_tokens, config_.vocab_size, config_.hidden_size,
        workspace_buffer_float, workspace_size, false, true
    );
}

// --- Explicit Template Instantiations ---
template class RMSNorm<__nv_bfloat16>;
template class L4maMlp<__nv_bfloat16>;
template class L4maAttention<__nv_bfloat16>;
template class L4maDecoderLayer<__nv_bfloat16>;
template class L4maModel<__nv_bfloat16>;
template class L4maForCausalLM<__nv_bfloat16>;
