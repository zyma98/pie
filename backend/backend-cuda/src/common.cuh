#pragma once

#include <thrust/device_vector.h>
#include <cublasLt.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <cstdint>
#include <cstdio>

// A simple CUDA error-checking macro for robust code.
#define CHECK_CUDA(call)                                                                               \
    do                                                                                                 \
    {                                                                                                  \
        cudaError_t err = call;                                                                        \
        if (err != cudaSuccess)                                                                        \
        {                                                                                              \
            fprintf(stderr, "CUDA Error: %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
            throw std::runtime_error(cudaGetErrorString(err));                                         \
        }                                                                                              \
    } while (0)

// Forward declare the datatypes from CUDA headers
struct __half;
struct __nv_bfloat16;

/**
 * @brief Host-side launch function for embedding lookup.
 *
 * This function provides a clean, library-style interface using thrust::device_vector
 * and an explicit CUDA stream for asynchronous execution. It assumes uint32_t indices.
 *
 * @tparam T The base data type (float, __half, __nv_bfloat16).
 * @param embedding The embedding matrix stored as a flat device vector.
 * @param indices A device vector of uint32_t indices to look up.
 * @param result A pointer to a device vector where the output will be stored.
 * @param embed_width The dimensionality of a single embedding vector.
 * @param stream The CUDA stream for asynchronous execution.
 */
template <typename T>
void embed(
    const thrust::device_vector<T> &embedding,
    const thrust::device_vector<uint32_t> &indices,
    thrust::device_vector<T> *result,
    int embed_width,
    cudaStream_t stream);

template <typename T>
void gemm_cublasLt(cublasLtHandle_t ltHandle,
                   cudaStream_t stream,
                   const thrust::device_vector<T> &A,
                   const thrust::device_vector<T> &B,
                   const thrust::device_vector<T> *bias,
                   thrust::device_vector<T> &C,
                   int m, int n, int k,
                   thrust::device_vector<char> &workspace,
                   bool transa,
                   bool transb);