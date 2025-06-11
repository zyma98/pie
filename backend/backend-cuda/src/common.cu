#include "common.cuh"

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdexcept>

/**
 * @brief High-performance CUDA kernel for embedding lookup.
 *
 * This version is specialized for uint32_t indices and uses 128-bit
 * vectorized memory operations for maximum bandwidth.
 *
 * @tparam T The base data type (float, __half, etc.).
 */
template <typename T>
__global__ void embedding_lookup_kernel_128bit(T *output,
                                               const T *embedding_matrix,
                                               const uint32_t *indices,
                                               int n,
                                               int hidden_dim_div_16)
{
    // Each block processes one lookup index.
    int idx_n = blockIdx.x;
    if (idx_n >= n)
    {
        return;
    }

    // Use shared memory to broadcast the source row index for the block.
    __shared__ uint32_t source_row_idx;
    if (threadIdx.x == 0)
    {
        source_row_idx = indices[idx_n];
    }
    __syncthreads();

    // Cast pointers to a 128-bit type (float4) to perform 16-byte memory transfers.
    // This is the core optimization, significantly increasing memory throughput.
    const float4 *source_row_ptr = reinterpret_cast<const float4 *>(embedding_matrix) + (long long)source_row_idx * hidden_dim_div_16;
    float4 *dest_row_ptr = reinterpret_cast<float4 *>(output) + (long long)idx_n * hidden_dim_div_16;

    // Use a grid-stride loop for threads to collectively copy the entire row.
    // This ensures that all data is copied regardless of the number of threads per block.
    for (int i = threadIdx.x; i < hidden_dim_div_16; i += blockDim.x)
    {
        dest_row_ptr[i] = source_row_ptr[i];
    }
}



/**
 * @brief Host-side launch function with a Thrust-based API.
 *
 * This function provides a clean, library-style interface using thrust::device_vector
 * and an explicit CUDA stream for asynchronous execution. It assumes uint32_t indices.
 *
 * @tparam T The base data type (float, __half, etc.).
 */
template <typename T>
void embed(
    const thrust::device_vector<T> &embedding,
    const thrust::device_vector<uint32_t> &indices,
    thrust::device_vector<T> *result,
    int embed_width,
    cudaStream_t stream)
{
    // --- Input Validation ---
    if (embedding.size() == 0 || indices.size() == 0) return;
    if (embedding.size() % embed_width != 0) {
        throw std::invalid_argument("Embedding vector size is not divisible by the embed_width.");
    }
    if ((embed_width * sizeof(T)) % 16 != 0) {
        throw std::invalid_argument("Total byte size of a slice (embed_width * sizeof(T)) must be a multiple of 16.");
    }
    
    // --- Prepare Parameters ---
    const int num_indices = indices.size();
    result->resize((long long)num_indices * embed_width);
    
    const int threads_per_block = 256;
    const int hidden_dim_div_16 = (embed_width * sizeof(T)) / 16;

    dim3 blocks(num_indices);
    dim3 threads(threads_per_block);
    
    // --- Kernel Launch ---
    embedding_lookup_kernel_128bit<T><<<blocks, threads, 0, stream>>>(
        thrust::raw_pointer_cast(result->data()),
        thrust::raw_pointer_cast(embedding.data()),
        thrust::raw_pointer_cast(indices.data()),
        num_indices,
        hidden_dim_div_16
    );
}

// --- Explicit Template Instantiations ---
// We explicitly instantiate the templates for the supported types. This forces
// the compiler to generate the code for each of these types, which will then
// be linked against when another file includes embedding.h.

template void embed<float>(
    const thrust::device_vector<float> &,
    const thrust::device_vector<uint32_t> &,
    thrust::device_vector<float> *,
    int, cudaStream_t);

template void embed<__half>(
    const thrust::device_vector<__half> &,
    const thrust::device_vector<uint32_t> &,
    thrust::device_vector<__half> *,
    int, cudaStream_t);

template void embed<__nv_bfloat16>(
    const thrust::device_vector<__nv_bfloat16> &,
    const thrust::device_vector<uint32_t> &,
    thrust::device_vector<__nv_bfloat16> *,
    int, cudaStream_t);
