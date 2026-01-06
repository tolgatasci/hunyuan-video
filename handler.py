"""
HunyuanVideo Multi-Model RunPod Serverless Handler
Supports: Avatar, I2V (Image-to-Video), T2V (Text-to-Video)

Usage:
  - mode: "avatar" | "i2v" | "t2v"
  - For avatar: image_url + audio_url
  - For i2v: image_url + prompt
  - For t2v: prompt only
"""

import os
import sys
import io
import base64
import time
import tempfile
import subprocess
import requests
import runpod
from pathlib import Path

# Globals
MODEL_BASE = os.environ.get("MODEL_BASE", "/runpod-volume/models/hunyuan")
TEMP_DIR = Path("/app/temp")
OUTPUT_DIR = Path("/app/output")

# Ensure directories exist
TEMP_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def download_file(url: str, save_path: Path) -> bool:
    """Download file from URL"""
    try:
        print(f"Downloading: {url}")
        response = requests.get(url, stream=True, timeout=300)
        response.raise_for_status()

        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        size_mb = save_path.stat().st_size / 1024 / 1024
        print(f"Downloaded: {save_path.name} ({size_mb:.1f}MB)")
        return True
    except Exception as e:
        print(f"Download error: {e}")
        return False


def download_models_if_needed(mode: str):
    """Download required model weights from HuggingFace"""
    from huggingface_hub import hf_hub_download, snapshot_download

    hf_token = os.environ.get("HF_TOKEN")

    model_path = Path(MODEL_BASE)
    model_path.mkdir(parents=True, exist_ok=True)

    if mode == "avatar":
        # Download HunyuanVideo-Avatar weights
        avatar_path = model_path / "hunyuan-avatar"
        if not avatar_path.exists():
            print("Downloading HunyuanVideo-Avatar model...")
            snapshot_download(
                repo_id="tencent/HunyuanVideo-Avatar",
                local_dir=str(avatar_path),
                token=hf_token
            )
            print("Avatar model downloaded!")

    elif mode in ["i2v", "t2v"]:
        # Download base HunyuanVideo weights
        base_path = model_path / "hunyuan-video"
        if not base_path.exists():
            print("Downloading HunyuanVideo model...")
            snapshot_download(
                repo_id="tencent/HunyuanVideo",
                local_dir=str(base_path),
                token=hf_token
            )
            print("Base model downloaded!")


def generate_avatar(image_path: Path, audio_path: Path, output_path: Path, **kwargs) -> dict:
    """Generate talking avatar video using HunyuanVideo-Avatar"""

    # Create CSV input file
    csv_path = TEMP_DIR / "avatar_input.csv"

    # CSV format for HunyuanVideo-Avatar
    csv_content = f"image_path,audio_path,text\n{image_path},{audio_path},A person speaking naturally"
    csv_path.write_text(csv_content)

    # Get parameters
    num_frames = kwargs.get("num_frames", 129)
    image_size = kwargs.get("image_size", 704)
    cfg_scale = kwargs.get("cfg_scale", 7.5)
    infer_steps = kwargs.get("infer_steps", 50)
    seed = kwargs.get("seed", 42)
    use_fp8 = kwargs.get("use_fp8", True)
    cpu_offload = kwargs.get("cpu_offload", False)

    # Build command
    checkpoint_path = f"{MODEL_BASE}/hunyuan-avatar/ckpts/hunyuan-video-t2v-720p/transformers/"
    if use_fp8:
        checkpoint_path += "mp_rank_00_model_states_fp8.pt"
    else:
        checkpoint_path += "mp_rank_00_model_states.pt"

    # MODEL_BASE should point to the directory containing ckpts/
    avatar_model_base = f"{MODEL_BASE}/hunyuan-avatar"

    cmd = [
        "python3", "/app/HunyuanVideo-Avatar/hymm_sp/sample_gpu_poor.py",
        "--input", str(csv_path),
        "--ckpt", checkpoint_path,
        "--sample-n-frames", str(num_frames),
        "--seed", str(seed),
        "--image-size", str(image_size),
        "--cfg-scale", str(cfg_scale),
        "--infer-steps", str(infer_steps),
        "--use-deepcache", "1",
        "--flow-shift-eval-video", "5.0",
        "--save-path", str(OUTPUT_DIR),
        "--infer-min"
    ]

    if use_fp8:
        cmd.append("--use-fp8")

    if cpu_offload:
        cmd.append("--cpu-offload")

    # Set environment - MODEL_BASE must point to directory containing ckpts/
    env = os.environ.copy()
    env["MODEL_BASE"] = avatar_model_base
    env["DISABLE_SP"] = "1"
    env["PYTHONPATH"] = "/app/HunyuanVideo-Avatar"
    if cpu_offload:
        env["CPU_OFFLOAD"] = "1"

    print(f"Running Avatar generation...")
    print(f"Command: {' '.join(cmd)}")

    start = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=1800)
    elapsed = time.time() - start

    if result.returncode != 0:
        print(f"STDERR: {result.stderr}")
        return {"error": f"Avatar generation failed: {result.stderr}"}

    # Find output video
    output_videos = list(OUTPUT_DIR.glob("*.mp4"))
    if not output_videos:
        return {"error": "No output video generated"}

    output_video = output_videos[-1]  # Get latest

    # Read and encode video
    with open(output_video, 'rb') as f:
        video_base64 = base64.b64encode(f.read()).decode('utf-8')

    return {
        "video_base64": video_base64,
        "duration": elapsed,
        "output_path": str(output_video)
    }


def generate_i2v(image_path: Path, prompt: str, output_path: Path, **kwargs) -> dict:
    """Generate video from image using HunyuanVideo-I2V"""

    # Get parameters
    resolution = kwargs.get("resolution", "720p")
    stability = kwargs.get("stability", True)
    flow_shift = kwargs.get("flow_shift", 7.0 if stability else 17.0)
    infer_steps = kwargs.get("infer_steps", 50)
    seed = kwargs.get("seed", 42)

    cmd = [
        "python3", "/app/HunyuanVideo-I2V/sample_image2video.py",
        "--model", "HYVideo-T/2",
        "--prompt", prompt,
        "--i2v-mode",
        "--i2v-image-path", str(image_path),
        "--i2v-resolution", resolution,
        "--flow-shift", str(flow_shift),
        "--infer-steps", str(infer_steps),
        "--seed", str(seed),
        "--use-cpu-offload",
        "--save-path", str(OUTPUT_DIR)
    ]

    if stability:
        cmd.append("--i2v-stability")

    env = os.environ.copy()
    env["PYTHONPATH"] = "/app/HunyuanVideo-I2V"

    print(f"Running I2V generation...")
    print(f"Prompt: {prompt}")

    start = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=3600)
    elapsed = time.time() - start

    if result.returncode != 0:
        print(f"STDERR: {result.stderr}")
        return {"error": f"I2V generation failed: {result.stderr}"}

    # Find output video
    output_videos = list(OUTPUT_DIR.glob("*.mp4"))
    if not output_videos:
        return {"error": "No output video generated"}

    output_video = output_videos[-1]

    with open(output_video, 'rb') as f:
        video_base64 = base64.b64encode(f.read()).decode('utf-8')

    return {
        "video_base64": video_base64,
        "duration": elapsed,
        "output_path": str(output_video)
    }


def generate_t2v(prompt: str, output_path: Path, **kwargs) -> dict:
    """Generate video from text using HunyuanVideo"""

    # Get parameters
    width = kwargs.get("width", 1280)
    height = kwargs.get("height", 720)
    video_length = kwargs.get("video_length", 129)
    infer_steps = kwargs.get("infer_steps", 50)
    cfg_scale = kwargs.get("cfg_scale", 6.0)
    seed = kwargs.get("seed", 42)

    cmd = [
        "python3", "/app/HunyuanVideo/sample_video.py",
        "--video-size", str(height), str(width),
        "--video-length", str(video_length),
        "--infer-steps", str(infer_steps),
        "--prompt", prompt,
        "--flow-reverse",
        "--use-cpu-offload",
        "--embedded-cfg-scale", str(cfg_scale),
        "--seed", str(seed),
        "--save-path", str(OUTPUT_DIR)
    ]

    env = os.environ.copy()
    env["PYTHONPATH"] = "/app/HunyuanVideo"

    print(f"Running T2V generation...")
    print(f"Prompt: {prompt}")
    print(f"Size: {width}x{height}, Frames: {video_length}")

    start = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=3600)
    elapsed = time.time() - start

    if result.returncode != 0:
        print(f"STDERR: {result.stderr}")
        return {"error": f"T2V generation failed: {result.stderr}"}

    # Find output video
    output_videos = list(OUTPUT_DIR.glob("*.mp4"))
    if not output_videos:
        return {"error": "No output video generated"}

    output_video = output_videos[-1]

    with open(output_video, 'rb') as f:
        video_base64 = base64.b64encode(f.read()).decode('utf-8')

    return {
        "video_base64": video_base64,
        "duration": elapsed,
        "output_path": str(output_video)
    }


def handler(job):
    """
    Main RunPod handler

    Input format:
    {
        "mode": "avatar" | "i2v" | "t2v",

        # For avatar mode:
        "image_url": "https://...",
        "audio_url": "https://...",

        # For i2v mode:
        "image_url": "https://...",
        "prompt": "A person walking...",

        # For t2v mode:
        "prompt": "A cat walks on grass...",

        # Optional parameters:
        "num_frames": 129,
        "width": 1280,
        "height": 720,
        "seed": 42,
        "use_fp8": true,
        "cpu_offload": false
    }
    """
    job_input = job.get("input", {})
    job_id = job.get("id", "unknown")

    mode = job_input.get("mode", "avatar")

    print("\n" + "=" * 60)
    print(f"HunyuanVideo Job: {job_id}")
    print(f"Mode: {mode}")
    print("=" * 60)

    try:
        # Download models if needed
        download_models_if_needed(mode)

        # Clean temp directory
        for f in TEMP_DIR.glob("*"):
            f.unlink()
        for f in OUTPUT_DIR.glob("*"):
            f.unlink()

        if mode == "avatar":
            # Avatar mode: image + audio → talking video
            # Supports both URL and base64 input
            image_url = job_input.get("image_url")
            audio_url = job_input.get("audio_url")
            image_base64 = job_input.get("image_base64")
            audio_base64 = job_input.get("audio_base64")

            image_path = TEMP_DIR / "input_image.jpg"
            audio_path = TEMP_DIR / "input_audio.wav"
            output_path = OUTPUT_DIR / "avatar_output.mp4"

            # Handle image input (URL or base64)
            if image_base64:
                print("  Using base64 image input")
                image_data = base64.b64decode(image_base64)
                with open(image_path, 'wb') as f:
                    f.write(image_data)
                print(f"  Image saved: {len(image_data)//1024}KB")
            elif image_url:
                if not download_file(image_url, image_path):
                    return {"error": "Failed to download image"}
            else:
                return {"error": "Avatar mode requires image_url or image_base64"}

            # Handle audio input (URL or base64)
            if audio_base64:
                print("  Using base64 audio input")
                audio_data = base64.b64decode(audio_base64)
                with open(audio_path, 'wb') as f:
                    f.write(audio_data)
                print(f"  Audio saved: {len(audio_data)//1024}KB")
            elif audio_url:
                if not download_file(audio_url, audio_path):
                    return {"error": "Failed to download audio"}
            else:
                return {"error": "Avatar mode requires audio_url or audio_base64"}

            result = generate_avatar(
                image_path, audio_path, output_path,
                num_frames=job_input.get("num_frames", 129),
                image_size=job_input.get("image_size", 704),
                cfg_scale=job_input.get("cfg_scale", 7.5),
                infer_steps=job_input.get("infer_steps", 50),
                seed=job_input.get("seed", 42),
                use_fp8=job_input.get("use_fp8", True),
                cpu_offload=job_input.get("cpu_offload", False)
            )

        elif mode == "i2v":
            # Image-to-Video mode
            image_url = job_input.get("image_url")
            prompt = job_input.get("prompt", "")

            if not image_url:
                return {"error": "I2V mode requires image_url"}

            image_path = TEMP_DIR / "input_image.jpg"
            output_path = OUTPUT_DIR / "i2v_output.mp4"

            if not download_file(image_url, image_path):
                return {"error": "Failed to download image"}

            result = generate_i2v(
                image_path, prompt, output_path,
                resolution=job_input.get("resolution", "720p"),
                stability=job_input.get("stability", True),
                flow_shift=job_input.get("flow_shift", 7.0),
                infer_steps=job_input.get("infer_steps", 50),
                seed=job_input.get("seed", 42)
            )

        elif mode == "t2v":
            # Text-to-Video mode
            prompt = job_input.get("prompt")

            if not prompt:
                return {"error": "T2V mode requires prompt"}

            output_path = OUTPUT_DIR / "t2v_output.mp4"

            result = generate_t2v(
                prompt, output_path,
                width=job_input.get("width", 1280),
                height=job_input.get("height", 720),
                video_length=job_input.get("video_length", 129),
                infer_steps=job_input.get("infer_steps", 50),
                cfg_scale=job_input.get("cfg_scale", 6.0),
                seed=job_input.get("seed", 42)
            )

        else:
            return {"error": f"Unknown mode: {mode}. Use 'avatar', 'i2v', or 't2v'"}

        print("=" * 60)
        print(f"Job completed: {job_id}")
        print("=" * 60 + "\n")

        return result

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("HunyuanVideo Multi-Model RunPod Worker")
    print("Modes: avatar | i2v | t2v")
    print("=" * 60)

    # Check GPU
    try:
        import torch
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"VRAM: {vram:.1f}GB")
        else:
            print("WARNING: CUDA not available!")
    except Exception as e:
        print(f"GPU check error: {e}")

    hf_token = os.environ.get("HF_TOKEN")
    print(f"HF_TOKEN: {'Set' if hf_token else 'NOT SET!'}")
    print("=" * 60 + "\n")

    runpod.serverless.start({"handler": handler})
