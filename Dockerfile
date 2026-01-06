# HunyuanVideo Multi-Model RunPod Serverless Worker
# Supports: Avatar (audio-driven), I2V (image-to-video), T2V (text-to-video)
# Optimized for 24GB-80GB VRAM GPUs

FROM hunyuanvideo/hunyuanvideo:cuda_12

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/runpod-volume/huggingface
ENV MODEL_BASE=/runpod-volume/models/hunyuan

WORKDIR /app

# Install additional dependencies
RUN pip install --no-cache-dir \
    runpod \
    huggingface_hub \
    cloudinary \
    requests \
    aiohttp \
    aiofiles

# Clone HunyuanVideo-Avatar for avatar support
RUN git clone https://github.com/Tencent-Hunyuan/HunyuanVideo-Avatar.git /app/HunyuanVideo-Avatar && \
    cd /app/HunyuanVideo-Avatar && \
    pip install --no-cache-dir -r requirements.txt || true

# Clone HunyuanVideo-I2V for image-to-video support
RUN git clone https://github.com/Tencent-Hunyuan/HunyuanVideo-I2V.git /app/HunyuanVideo-I2V && \
    cd /app/HunyuanVideo-I2V && \
    pip install --no-cache-dir -r requirements.txt || true

# Copy handler
COPY handler.py /app/handler.py

# Create directories
RUN mkdir -p /app/temp /app/output

CMD ["python3", "-u", "handler.py"]
