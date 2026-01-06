#!/bin/bash

# HunyuanVideo Multi-Model Docker Build Script
# Supports: Avatar, I2V, T2V
# Usage: ./build.sh [DOCKERHUB_USERNAME]

set -e

DOCKERHUB_USER=${1:-"tolgatasci"}
IMAGE_NAME="hunyuan-video"
VERSION="1.0"

echo "========================================"
echo "HunyuanVideo Multi-Model Docker Build"
echo "========================================"
echo ""
echo "Image: ${DOCKERHUB_USER}/${IMAGE_NAME}:${VERSION}"
echo ""
echo "This image supports:"
echo "  - Avatar: Audio-driven talking head (10-24GB VRAM)"
echo "  - I2V: Image-to-Video (48-60GB VRAM)"
echo "  - T2V: Text-to-Video (48-60GB VRAM)"
echo ""
echo "Base image: hunyuanvideo/hunyuanvideo:cuda_12"
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 1
fi

# Build
echo ""
echo "Building Docker image..."
docker build -t ${DOCKERHUB_USER}/${IMAGE_NAME}:${VERSION} .

echo ""
echo "Build complete!"
echo ""

# Push?
read -p "Push to Docker Hub? (y/n) " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Pushing to Docker Hub..."
    docker push ${DOCKERHUB_USER}/${IMAGE_NAME}:${VERSION}
    echo ""
    echo "========================================"
    echo "SUCCESS!"
    echo "========================================"
    echo ""
    echo "Container Image: ${DOCKERHUB_USER}/${IMAGE_NAME}:${VERSION}"
    echo ""
    echo "RunPod Serverless Setup:"
    echo "========================"
    echo ""
    echo "1. Go to https://console.runpod.io/serverless"
    echo "2. Create New Endpoint"
    echo "3. Container Image: ${DOCKERHUB_USER}/${IMAGE_NAME}:${VERSION}"
    echo ""
    echo "GPU Requirements:"
    echo "  - Avatar mode: 24GB+ (RTX 4090, A10, L40S)"
    echo "  - I2V/T2V mode: 48GB+ (A100, H100)"
    echo ""
    echo "Environment Variables:"
    echo "  - HF_TOKEN: Your HuggingFace token"
    echo ""
    echo "Volume:"
    echo "  - Mount Network Volume to /runpod-volume"
    echo "  - Models will be cached there"
    echo ""
    echo "Example API Call (Avatar):"
    echo '  {"input": {'
    echo '    "mode": "avatar",'
    echo '    "image_url": "https://example.com/face.jpg",'
    echo '    "audio_url": "https://example.com/speech.wav"'
    echo '  }}'
    echo ""
    echo "Example API Call (I2V):"
    echo '  {"input": {'
    echo '    "mode": "i2v",'
    echo '    "image_url": "https://example.com/scene.jpg",'
    echo '    "prompt": "The person starts walking"'
    echo '  }}'
    echo ""
    echo "Example API Call (T2V):"
    echo '  {"input": {'
    echo '    "mode": "t2v",'
    echo '    "prompt": "A cat walks on grass, realistic style",'
    echo '    "width": 1280,'
    echo '    "height": 720'
    echo '  }}'
else
    echo ""
    echo "Image built locally: ${DOCKERHUB_USER}/${IMAGE_NAME}:${VERSION}"
    echo "Run 'docker push ${DOCKERHUB_USER}/${IMAGE_NAME}:${VERSION}' to push later."
fi
