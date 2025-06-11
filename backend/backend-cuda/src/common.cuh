// allocation_options.cuh

#pragma once

#include <thrust/device_vector.h>

template <typename ElemT, typename IndexT, typename OffsetT>
void embed(
    const thrust::device_vector<ElemT> &embedding,
    const thrust::device_vector<IndexT> &indices,
    const thrust::device_vector<OffsetT> &offsets,
    thrust::device_vector<ElemT> *result,
    int batch_size,
    int embed_width,
    cudaStream_t stream = 0);