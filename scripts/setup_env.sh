#!/bin/bash
set -e

echo "Setting up TokenSmith environment (Conda-only dependencies)..."

# Detect platform
OS=$(uname -s)
ARCH=$(uname -m)

echo "Detected: $OS $ARCH"

detect_amd_gpu() {
    if command -v rocm-smi &> /dev/null; then
        return 0
    fi
    if command -v lspci &> /dev/null && lspci | grep -Eiq 'vga|3d|display'; then
        lspci | grep -Eiq 'amd|radeon|advanced micro devices'
        return $?
    fi
    return 1
}

# Platform-specific CMAKE_ARGS for llama-cpp-python
if [[ "$OS" == "Darwin" ]]; then
    if [[ "$ARCH" == "arm64" ]]; then
        echo "Apple Silicon detected - enabling Metal support"
        export CMAKE_ARGS="-DGGML_METAL=on -DGGML_ACCELERATE=on"
        export FORCE_CMAKE=1
    fi
elif [[ "$OS" == "Linux" ]]; then
    if command -v nvidia-smi &> /dev/null; then
        echo "NVIDIA GPU detected - enabling CUDA support"
        export CMAKE_ARGS="-DGGML_CUDA=on"
        export FORCE_CMAKE=1
    elif detect_amd_gpu; then
        echo "AMD GPU detected - enabling Vulkan support"
        export CMAKE_ARGS="-DGGML_VULKAN=on"
        export FORCE_CMAKE=1
    else
        export CMAKE_ARGS="-DGGML_BLAS=on -DGGML_BLAS_VENDOR=OpenBLAS"
    fi
fi

# Install llama-cpp-python with platform-specific optimizations
# (This is one of the few packages that needs pip due to compilation flags)
if [[ -n "$CMAKE_ARGS" ]]; then
    echo "Installing llama-cpp-python with: $CMAKE_ARGS"
    CMAKE_ARGS="$CMAKE_ARGS" pip install llama-cpp-python --force-reinstall --no-cache-dir
else
    pip install llama-cpp-python
fi

echo "TokenSmith environment setup complete!"
echo "All dependencies managed by Conda."
