

### Prerequisites for setting up vLLM/SGLang/Llama.cpp

install docker

Install NVIDIA Container Toolkit,
following the instructions below
https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html





# dependencies
openai




# Installing flashinfer
```
# Set target CUDA architectures
export TORCH_CUDA_ARCH_LIST="7.5 8.0 8.9 9.0a 10.0a"
# Build AOT kernels. Will produce AOT kernels in aot-ops/
python -m flashinfer.aot
# Build AOT wheel
python -m build --no-isolation --wheel
# Install AOT wheel
python -m pip install dist/flashinfer-*.whl
```


pip install py_mini_racer
pip install sglang
pip install pybase64
blake3