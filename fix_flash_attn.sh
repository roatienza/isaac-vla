#!/bin/bash
# =============================================================================
# Fix flash-attn CUDA mismatch for RTX 5090 + OpenVLA-OFT
# =============================================================================
# Problem: System CUDA 13.1 mismatches PyTorch cu121
# Solution: Rebuild env with matching CUDA versions
# =============================================================================

set -e

echo "============================================"
echo "Step 0: Check current CUDA/driver versions"
echo "============================================"
nvidia-smi | head -5
echo ""
echo "System nvcc version:"
nvcc --version 2>/dev/null || echo "nvcc not found in PATH"
echo ""

echo "============================================"
echo "Step 1: Remove old environment and recreate"
echo "============================================"
conda deactivate 2>/dev/null || true
conda remove -n vla-oft --all -y 2>/dev/null || true
conda create -n vla-oft python=3.10 -y
conda activate vla-oft

echo "============================================"
echo "Step 2: Install PyTorch with CUDA 12.4+"
echo "============================================"
# RTX 5090 (Blackwell) needs CUDA 12.4+ for compute capability 10.0
# PyTorch 2.5+ supports CUDA 12.4
# Adjust the CUDA version below based on what PyTorch offers:
#   - For PyTorch 2.5: cu124
#   - For PyTorch 2.6+: cu124 or cu126 if available
#
# Check https://pytorch.org/get-started/locally/ for latest

# Option A: PyTorch with CUDA 12.4 (recommended for RTX 5090)
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu124

# Verify PyTorch sees CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'GPU: {torch.cuda.get_device_name(0)}' if torch.cuda.is_available() else 'No GPU')"

echo "============================================"
echo "Step 3: Install CUDA toolkit matching PyTorch"
echo "============================================"
# This ensures flash-attn builds against the SAME CUDA as PyTorch
# nvidia-cuda-toolkit installs the CUDA headers/libs that match PyTorch's cu124
pip install nvidia-cuda-toolkit==12.4.1

# Set CUDA_HOME to the pip-installed toolkit (NOT the system /usr/local/cuda)
CUDA_TOOLKIT_PATH=$(python -c "import nvidia.cuda; import os; print(os.path.dirname(nvidia.cuda.__file__))")
export CUDA_HOME="$CUDA_TOOLKIT_PATH"
export CUDA_PATH="$CUDA_TOOLKIT_PATH"
export LD_LIBRARY_PATH="$CUDA_TOOLKIT_PATH/lib:$LD_LIBRARY_PATH"

echo "CUDA_HOME set to: $CUDA_HOME"
echo ""

echo "============================================"
echo "Step 4: Install flash-attn"
echo "============================================"
# Set TMPDIR to same filesystem as pip cache (fixes errno 18)
export TMPDIR="$HOME/.cache/pip/tmp"
mkdir -p "$TMPDIR"

# Try prebuilt wheel first (fastest)
echo "Attempting prebuilt wheel installation..."
pip install flash-attn --no-build-isolation 2>/dev/null || {
    echo "Prebuilt wheel not available, building from source..."
    # Build from source using the matching CUDA toolkit
    pip install flash-attn --no-build-isolation \
        --env TMPDIR="$TMPDIR" \
        --env CUDA_HOME="$CUDA_HOME"
}

echo "============================================"
echo "Step 5: Verify flash-attn"
echo "============================================"
python -c "
import flash_attn
print(f'flash-attn version: {flash_attn.__version__}')
import torch
print(f'PyTorch CUDA: {torch.version.cuda}')
print(f'flash-attn CUDA: {flash_attn.cuda}')
print('flash-attn installed successfully!')
"

echo "============================================"
echo "Step 6: Install OpenVLA-OFT and remaining deps"
echo "============================================"
cd /tmp
if [ ! -d openvla-oft ]; then
    git clone https://github.com/moojink/openvla-oft.git
fi
cd openvla-oft
pip install -e .

echo ""
echo "============================================"
echo "Step 7: Final verification"
echo "============================================"
python -c "
import torch
import flash_attn
import transformers
print('All imports successful!')
print(f'  PyTorch:    {torch.__version__} (CUDA {torch.version.cuda})')
print(f'  flash-attn: {flash_attn.__version__}')
print(f'  GPU:        {torch.cuda.get_device_name(0)}')
print(f'  VRAM:       {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
"