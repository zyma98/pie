#pragma once

#include <vector>
#include <memory>
#include <type_traits> // Required for std::is_same_v
#include <cuda_runtime.h>
#include <cuda_bf16.h>

template <typename T>
class Tensor {
public:
    // --- Constructors and Lifetime ---
    explicit Tensor(size_t size);
    Tensor(const Tensor& other, size_t offset); // View constructor

    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;

    Tensor(Tensor&&) = default;
    Tensor& operator=(Tensor&&) = default;

    ~Tensor() = default;


    // --- Accessors ---
    size_t size() const;
    T* data() const;


    // --- Host/Device Data Transfer ---
    std::vector<T> to_vector() const;
    void from_vector(const std::vector<T>& data);
    void from_pointer(const T* data, size_t count);


    // --- Debugging Methods ---
    // This function will only exist for instantiations where T is float or __nv_bfloat16
    float mean() const requires std::is_same_v<T, float> || std::is_same_v<T, __nv_bfloat16>;

    void print(size_t start_index = 0, size_t count = -1) const;


private:
    struct CudaDeleter {
        void operator()(void* ptr) const;
    };

    std::shared_ptr<T> data_;
    size_t size_;
    size_t offset_;
};


extern template class Tensor<float>;
extern template class Tensor<__nv_bfloat16>;
extern template class Tensor<int32_t>;
extern template class Tensor<uint32_t>;
extern template class Tensor<uint8_t>;