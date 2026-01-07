#!/bin/bash
# HunyuanVideo Startup Script
# Sets up model symlinks and starts the handler

echo "=============================================="
echo "HunyuanVideo Multi-Model Worker Starting..."
echo "=============================================="

MODEL_BASE="${MODEL_BASE:-/runpod-volume/models/hunyuan}"

# Create model directories
mkdir -p "$MODEL_BASE"
mkdir -p /app/temp /app/output

# Check and create symlinks for I2V models
echo "Checking model paths..."

# Check if Avatar ckpts exist (these contain base models)
if [ -d "$MODEL_BASE/hunyuan-avatar/ckpts" ]; then
    echo "  Found Avatar models at: $MODEL_BASE/hunyuan-avatar/ckpts"

    # Create symlink for I2V to use Avatar models
    if [ ! -e "$MODEL_BASE/ckpts" ]; then
        ln -s "$MODEL_BASE/hunyuan-avatar/ckpts" "$MODEL_BASE/ckpts"
        echo "  Created symlink: $MODEL_BASE/ckpts -> $MODEL_BASE/hunyuan-avatar/ckpts"
    fi
fi

# Check if HunyuanVideo base models exist
if [ -d "$MODEL_BASE/hunyuan-video/ckpts" ]; then
    echo "  Found HunyuanVideo models at: $MODEL_BASE/hunyuan-video/ckpts"

    if [ ! -e "$MODEL_BASE/ckpts" ]; then
        ln -s "$MODEL_BASE/hunyuan-video/ckpts" "$MODEL_BASE/ckpts"
        echo "  Created symlink: $MODEL_BASE/ckpts -> $MODEL_BASE/hunyuan-video/ckpts"
    fi
fi

# List available models
echo ""
echo "Available models:"
ls -la "$MODEL_BASE" 2>/dev/null || echo "  No models found yet"
echo ""

# Check GPU
python3 -c "
import torch
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f'GPU: {gpu_name} ({vram:.1f}GB VRAM)')
else:
    print('WARNING: No GPU detected!')
" 2>/dev/null || echo "GPU check failed"

echo ""
echo "Starting handler..."
echo "=============================================="

# Start the handler
exec python3 -u /app/handler.py
