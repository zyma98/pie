#include "l4ma.cuh"
#include <cublas_v2.h>
#include <cublasLt.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <stdexcept>
#include <cassert>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <flashinfer/activation.cuh>

// Macro for cuBLAS error checking
#define CUBLAS_CHECK(status) \
    do { \
        cublasStatus_t _status = (status); \
        if (_status != CUBLAS_STATUS_SUCCESS) { \
            printf("cuBLAS error at %s:%d: %d\n", __FILE__, __LINE__, _status); \
            throw std::runtime_error("cuBLAS error"); \
        } \
    } while (0)


__device__ __forceinline__ float silu(const float& val) { return val / (1.0f + __expf(-val)); }


template <typename T>
void silu_and_mul(
    thrust::device_vector<T>& out,
    const thrust::device_vector<T>& input,
    int num_tokens,
    int d,
    cudaStream_t stream,
    bool enable_pdl = false
) {
    T* out_ptr = thrust::raw_pointer_cast(out.data());
    const T* input_ptr = thrust::raw_pointer_cast(input.data());
    uint32_t vec_size = 16 / sizeof(T);
    cudaLaunchConfig_t config;
    config.gridDim = num_tokens;
    config.blockDim = std::min(d / vec_size, 1024U);
    config.dynamicSmemBytes = 0;
    config.stream = stream;
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[0].val.programmaticStreamSerializationAllowed = enable_pdl;
    config.numAttrs = 1;
    config.attrs = attrs;

    auto kernel = flashinfer::activation::act_and_mul_kernel<T, silu>;
    cudaLaunchKernelEx(&config, kernel, out_ptr, input_ptr, d);

}

// Helper for GEMM with bias using cublasLt, supporting float, half, and bf16
inline void gemm_bias_cublasLt(
    cublasLtHandle_t ltHandle,
    cudaStream_t stream,
    const void* A, // [m, k]
    const void* B, // [n, k] (will be transposed)
    const void* bias, // [n] or nullptr
    void* C, // [m, n]
    int m, int n, int k,
    cudaDataType_t dtype, // CUDA_R_32F, CUDA_R_16F, CUDA_R_16BF
    cublasComputeType_t computeType // CUBLAS_COMPUTE_32F, CUBLAS_COMPUTE_16F, CUBLAS_COMPUTE_16BF
) {
    cublasLtMatmulDesc_t matmulDesc;
    cublasLtMatrixLayout_t Adesc, Bdesc, Cdesc;
    float alpha = 1.0f, beta = 0.0f;
    cublasLtEpilogue_t epilogue = bias ? CUBLASLT_EPILOGUE_BIAS : CUBLASLT_EPILOGUE_DEFAULT;

    CUBLAS_CHECK(cublasLtMatmulDescCreate(&matmulDesc, computeType, dtype));
    cublasOperation_t opA = CUBLAS_OP_N;
    cublasOperation_t opB = CUBLAS_OP_T;
    cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSA, &opA, sizeof(cublasOperation_t));
    cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSB, &opB, sizeof(cublasOperation_t));
    cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue));
    if (bias) {
        cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias));
    }
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&Adesc, dtype, m, k, m));
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&Bdesc, dtype, n, k, n));
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&Cdesc, dtype, m, n, m));

    CUBLAS_CHECK(cublasLtMatmul(
        ltHandle,
        matmulDesc,
        &alpha,
        A, Adesc,
        B, Bdesc,
        &beta,
        C, Cdesc,
        C, Cdesc,
        nullptr, nullptr, 0, stream));

    cublasLtMatmulDescDestroy(matmulDesc);
    cublasLtMatrixLayoutDestroy(Adesc);
    cublasLtMatrixLayoutDestroy(Bdesc);
    cublasLtMatrixLayoutDestroy(Cdesc);
}



// L4maMlp implementation
L4maMlp<float>::L4maMlp(const L4maConfig &config,
        const thrust::device_vector<float> &gate_proj_weights,
        const thrust::device_vector<float> &up_proj_weights,
        const thrust::device_vector<float> &down_proj_weights,
        std::optional<thrust::device_vector<float>> gate_proj_bias,
        std::optional<thrust::device_vector<float>> up_proj_bias,
        std::optional<thrust::device_vector<float>> down_proj_bias)
    : config_(config),
      gate_proj_weights_(gate_proj_weights),
      up_proj_weights_(up_proj_weights),
      down_proj_weights_(down_proj_weights),
      gate_proj_bias_(std::move(gate_proj_bias)),
      up_proj_bias_(std::move(up_proj_bias)),
      down_proj_bias_(std::move(down_proj_bias))
{
    cublasCreate(&cublas_handle_);
    cublasLtCreate(&cublaslt_handle_);
}

L4maMlp<float>::~L4maMlp() {
    cublasDestroy(cublas_handle_);
    cublasLtDestroy(cublaslt_handle_);
}

void L4maMlp<float>::forward(
    thrust::device_vector<float> &output,
    const thrust::device_vector<float> &x,
    int num_tokens,
    thrust::device_vector<float> &temp_buffer_mlp,
    cudaStream_t stream)
{
    // Shapes:
    // x: [num_tokens, hidden_size]
    // gate_proj_weights_: [intermediate_size, hidden_size]
    // up_proj_weights_: [intermediate_size, hidden_size]
    // down_proj_weights_: [hidden_size, intermediate_size]
    // output: [num_tokens, hidden_size]
    // temp_buffer_mlp: must be at least 2 * num_tokens * intermediate_size + num_tokens * hidden_size

    assert(x.size() == num_tokens * config_.hidden_size);
    assert(output.size() == num_tokens * config_.hidden_size);
    int hs = config_.hidden_size;
    int is = config_.intermediate_size;

    const float* x_ptr = thrust::raw_pointer_cast(x.data());
    float* gate_proj_ptr = thrust::raw_pointer_cast(temp_buffer_mlp.data()); // [num_tokens, intermediate_size]
    float* up_proj_ptr = gate_proj_ptr + num_tokens * is; // [num_tokens, intermediate_size]
    float* silu_ptr = up_proj_ptr + num_tokens * is; // [num_tokens, intermediate_size]
    float* mul_ptr = silu_ptr; // reuse silu_ptr for mul result
    float* out_ptr = thrust::raw_pointer_cast(output.data());

    // 1. gate_proj = x * W_g^T (+ b_g)
    gemm_bias_cublasLt(
        cublaslt_handle_, stream,
        x_ptr, thrust::raw_pointer_cast(gate_proj_weights_.data()),
        gate_proj_bias_ ? thrust::raw_pointer_cast(gate_proj_bias_->data()) : nullptr,
        gate_proj_ptr,
        num_tokens, is, hs,
        CUDA_R_32F, CUBLAS_COMPUTE_32F);

    // 2. up_proj = x * W_u^T (+ b_u)
    gemm_bias_cublasLt(
        cublaslt_handle_, stream,
        x_ptr, thrust::raw_pointer_cast(up_proj_weights_.data()),
        up_proj_bias_ ? thrust::raw_pointer_cast(up_proj_bias_->data()) : nullptr,
        up_proj_ptr,
        num_tokens, is, hs,
        CUDA_R_32F, CUBLAS_COMPUTE_32F);

    // 3+4. Fused SiLU activation and elementwise multiply using flashinfer kernel
    silu_and_mul<float>(
        temp_buffer_mlp, // out: silu_ptr (same as mul_ptr)
        temp_buffer_mlp, // input: concat(gate_proj_ptr, up_proj_ptr)
        num_tokens,
        is,
        stream
    );

    // 5. Down projection: output = mul_ptr * W_d^T (+ b_d)
    gemm_bias_cublasLt(
        cublaslt_handle_, stream,
        mul_ptr, thrust::raw_pointer_cast(down_proj_weights_.data()),
        down_proj_bias_ ? thrust::raw_pointer_cast(down_proj_bias_->data()) : nullptr,
        out_ptr,
        num_tokens, hs, is,
        CUDA_R_32F, CUBLAS_COMPUTE_32F);
}
