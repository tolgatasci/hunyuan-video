# HunyuanVideo RunPod Serverless Worker

Multi-model RunPod serverless worker supporting:

| Mode | Description | VRAM |
|------|-------------|------|
| `avatar` | Audio-driven talking head | 10-24GB |
| `i2v` | Image-to-Video | 48-60GB |
| `t2v` | Text-to-Video | 48-60GB |

## Build & Deploy

```bash
# Build Docker image
./build.sh tolgatasci

# Or manually
docker build -t tolgatasci/hunyuan-video:1.0 .
docker push tolgatasci/hunyuan-video:1.0
```

## RunPod Setup

1. Go to [RunPod Serverless](https://console.runpod.io/serverless)
2. Create New Endpoint
3. Settings:
   - **Container Image**: `tolgatasci/hunyuan-video:1.0`
   - **GPU**: 24GB+ for Avatar, 48GB+ for I2V/T2V
   - **Environment Variables**:
     - `HF_TOKEN`: Your HuggingFace token
   - **Volume**: Mount to `/runpod-volume` (for model caching)

## API Usage

### Avatar Mode (Audio-driven talking head)

```json
{
  "input": {
    "mode": "avatar",
    "image_url": "https://example.com/face.jpg",
    "audio_url": "https://example.com/speech.wav",
    "num_frames": 129,
    "image_size": 704,
    "use_fp8": true,
    "cpu_offload": false
  }
}
```

### I2V Mode (Image-to-Video)

```json
{
  "input": {
    "mode": "i2v",
    "image_url": "https://example.com/scene.jpg",
    "prompt": "The person starts walking slowly",
    "resolution": "720p",
    "stability": true,
    "infer_steps": 50
  }
}
```

### T2V Mode (Text-to-Video)

```json
{
  "input": {
    "mode": "t2v",
    "prompt": "A cat walks on grass, realistic style",
    "width": 1280,
    "height": 720,
    "video_length": 129,
    "infer_steps": 50
  }
}
```

## Response Format

```json
{
  "video_base64": "base64_encoded_video...",
  "duration": 45.2,
  "output_path": "/app/output/avatar_output.mp4"
}
```

## Parameters

### Avatar Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_frames` | 129 | Number of video frames |
| `image_size` | 704 | Image resolution |
| `cfg_scale` | 7.5 | Guidance scale |
| `infer_steps` | 50 | Inference steps |
| `use_fp8` | true | Use FP8 quantization (saves VRAM) |
| `cpu_offload` | false | Enable CPU offloading |

### I2V Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| `resolution` | "720p" | Output resolution |
| `stability` | true | Stable video generation |
| `flow_shift` | 7.0 | Flow shift (17.0 for dynamic) |
| `infer_steps` | 50 | Inference steps |

### T2V Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| `width` | 1280 | Video width |
| `height` | 720 | Video height |
| `video_length` | 129 | Number of frames |
| `cfg_scale` | 6.0 | Guidance scale |
| `infer_steps` | 50 | Inference steps |
