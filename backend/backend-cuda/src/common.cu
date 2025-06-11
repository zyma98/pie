// allocation_options.cu

#include "common.cuh" // Or the appropriate path to your header file
#include "cuembed/include/embedding_lookup.cuh"

template <typename ElemT, typename IndexT, typename OffsetT>
void embed(
    const thrust::device_vector<ElemT> &embedding,
    const thrust::device_vector<IndexT> &indices,
    const thrust::device_vector<OffsetT> &offsets,
    thrust::device_vector<ElemT> *result,
    int batch_size,
    int embed_width,
    cudaStream_t stream)
{
    using InputT = ElemT;
    using OutputT = ElemT;
    EmbeddingForward<InputT, OutputT, IndexT, OffsetT>(
        embedding.data().get(),
        embed_width,
        indices.data().get(),
        nullptr,
        nullptr,
        batch_size,
        0,
        cuembed::CombineMode::kConcat,
        result->data().get(),
        stream);
}

// Explicit template instantiations if needed, e.g.:
// template void RunForward<float, int, int, false>(...);