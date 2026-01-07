"""
HunyuanVideo Multi-Model RunPod Serverless Handler
Supports: Avatar, I2V (Image-to-Video), T2V (Text-to-Video), Disk Management

Usage:
  - mode: "avatar" | "i2v" | "t2v" | "disk_check" | "cleanup"
  - For avatar: image_url/image_base64 + audio_url/audio_base64
  - For i2v: image_url/image_base64 + prompt
  - For t2v: prompt only
  - For disk_check: no params needed
  - For cleanup: optional clear_hf_cache=true
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
import shutil
import gc
from pathlib import Path

# Globals
MODEL_BASE = os.environ.get("MODEL_BASE", "/runpod-volume/models/hunyuan")
TEMP_DIR = Path("/app/temp")
OUTPUT_DIR = Path("/app/output")
HF_CACHE_DIR = Path.home() / ".cache" / "huggingface"

# Ensure directories exist
TEMP_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def get_disk_usage():
    """Get disk usage info"""
    try:
        total, used, free = shutil.disk_usage("/")
        return {
            "total_gb": total / (1024**3),
            "used_gb": used / (1024**3),
            "free_gb": free / (1024**3),
            "used_percent": (used / total) * 100
        }
    except Exception as e:
        print(f"Disk usage check error: {e}")
        return None


def cleanup_temp_files():
    """Clean up temporary files"""
    try:
        # Clean temp and output directories
        for f in TEMP_DIR.glob("*"):
            f.unlink()
        for f in OUTPUT_DIR.glob("*"):
            f.unlink()

        # Clean Python garbage
        gc.collect()

        # Try to free GPU memory
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except:
            pass

        print("  Cleanup completed")
    except Exception as e:
        print(f"  Cleanup error: {e}")


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

    elif mode == "i2v":
        # HunyuanVideo-I2V requires 3 separate model downloads:
        # 1. tencent/HunyuanVideo-I2V → ckpts/ (main I2V model)
        # 2. xtuner/llava-llama-3-8b-v1_1-transformers → ckpts/text_encoder_i2v/ (LLaVA text encoder)
        # 3. openai/clip-vit-large-patch14 → ckpts/text_encoder_2/ (CLIP encoder)
        #
        # Expected structure after download:
        # MODEL_BASE/ckpts/
        # ├── hunyuan-video-i2v-720p/transformers/mp_rank_00_model_states.pt
        # ├── text_encoder_i2v/  (LLaVA)
        # └── text_encoder_2/    (CLIP)

        # Clean up old incorrect folder structure if exists
        old_i2v_path = model_path / "hunyuan-i2v"
        if old_i2v_path.exists():
            print(f"Removing old incorrect I2V folder: {old_i2v_path}")
            import shutil
            shutil.rmtree(old_i2v_path)

        ckpts_path = model_path / "ckpts"
        i2v_model_file = ckpts_path / "hunyuan-video-i2v-720p" / "transformers" / "mp_rank_00_model_states.pt"
        text_encoder_i2v = ckpts_path / "text_encoder_i2v"
        text_encoder_2 = ckpts_path / "text_encoder_2"

        # Download main I2V model
        if not i2v_model_file.exists():
            print("Downloading HunyuanVideo-I2V main model from tencent/HunyuanVideo-I2V...")
            ckpts_path.mkdir(parents=True, exist_ok=True)
            snapshot_download(
                repo_id="tencent/HunyuanVideo-I2V",
                local_dir=str(ckpts_path),
                token=False  # Public repo - explicitly disable token to avoid cached expired token
            )
            print("I2V main model downloaded!")

        # Download LLaVA text encoder
        if not text_encoder_i2v.exists():
            print("Downloading LLaVA text encoder from xtuner/llava-llama-3-8b-v1_1-transformers...")
            snapshot_download(
                repo_id="xtuner/llava-llama-3-8b-v1_1-transformers",
                local_dir=str(text_encoder_i2v),
                token=False  # Public repo - explicitly disable token
            )
            print("LLaVA text encoder downloaded!")

        # Download CLIP text encoder
        if not text_encoder_2.exists():
            print("Downloading CLIP text encoder from openai/clip-vit-large-patch14...")
            snapshot_download(
                repo_id="openai/clip-vit-large-patch14",
                local_dir=str(text_encoder_2),
                token=False  # Public repo - explicitly disable token
            )
            print("CLIP text encoder downloaded!")

    elif mode == "t2v":
        # Download base HunyuanVideo T2V weights
        t2v_ckpts = model_path / "ckpts" / "hunyuan-video-t2v-720p"
        if not t2v_ckpts.exists():
            print("Downloading HunyuanVideo T2V model...")
            snapshot_download(
                repo_id="tencent/HunyuanVideo",
                local_dir=str(model_path),
                token=hf_token
            )
            print("T2V model downloaded!")


def generate_avatar(image_path: Path, audio_path: Path, output_path: Path, **kwargs) -> dict:
    """Generate talking avatar video using HunyuanVideo-Avatar"""

    # Create CSV input file
    csv_path = TEMP_DIR / "avatar_input.csv"

    # CSV format for HunyuanVideo-Avatar: videoid,image,audio,prompt,fps
    csv_content = f"videoid,image,audio,prompt,fps\n1,{image_path},{audio_path},A person speaking naturally,25"
    csv_path.write_text(csv_content)

    # Get parameters from request (passed by handler)
    num_frames = kwargs.get("num_frames", 129)
    image_size = kwargs.get("image_size", 704)  # 704 for quality, 512 for speed
    cfg_scale = kwargs.get("cfg_scale", 7.5)
    infer_steps = kwargs.get("infer_steps", 50)  # 50 for quality, 30 for speed
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
    print(f"  image_size={image_size}, infer_steps={infer_steps}, num_frames={num_frames}")
    print(f"  use_fp8={use_fp8}, cpu_offload={cpu_offload}")
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


def setup_i2v_models():
    """Setup I2V models - check if they exist at expected location"""
    # I2V models downloaded to: MODEL_BASE/ckpts/
    # Structure:
    #   MODEL_BASE/ckpts/hunyuan-video-i2v-720p/transformers/...
    #   MODEL_BASE/ckpts/text_encoder_i2v/
    #   MODEL_BASE/ckpts/text_encoder_2/
    #
    # IMPORTANT: The script's constants.py uses MODEL_BASE env var for paths:
    #   MODEL_BASE/text_encoder_i2v, MODEL_BASE/text_encoder_2
    # So we return MODEL_BASE/ckpts (not MODEL_BASE)

    ckpts_path = Path(MODEL_BASE) / "ckpts"
    model_file = ckpts_path / "hunyuan-video-i2v-720p" / "transformers" / "mp_rank_00_model_states.pt"
    text_encoder_i2v = ckpts_path / "text_encoder_i2v"
    text_encoder_2 = ckpts_path / "text_encoder_2"

    if model_file.exists() and text_encoder_i2v.exists() and text_encoder_2.exists():
        print(f"  I2V models found at: {ckpts_path}")
        return str(ckpts_path)

    print(f"  I2V models NOT complete. Will be downloaded by download_models_if_needed()")
    return str(ckpts_path)


def generate_i2v(image_path: Path, prompt: str, output_path: Path, **kwargs) -> dict:
    """Generate video from image using HunyuanVideo-I2V

    Optimized for 3-second video generation (75 frames @ 25fps)
    """

    # Get parameters - defaults optimized for 3-second Shorts
    # Note: (video_length - 1) must be multiple of 4, so valid values: 73, 77, 81, 85...
    resolution = kwargs.get("resolution", "540p")  # 540p for speed, 720p for quality
    num_frames = kwargs.get("num_frames", 77)  # ~3 seconds @ 25fps (77-1=76, divisible by 4)
    stability = kwargs.get("stability", True)
    flow_shift = kwargs.get("flow_shift", 7.0 if stability else 17.0)
    infer_steps = kwargs.get("infer_steps", 30)  # 30 for speed, 50 for quality
    seed = kwargs.get("seed", 42)
    fps = kwargs.get("fps", 25)

    # Setup and find models
    models_root = setup_i2v_models()

    if not models_root or not os.path.exists(models_root):
        # Last resort - check all possible locations
        # Note: models_root should be the ckpts folder itself (contains text_encoder_i2v, etc.)
        possible_paths = [
            f"{MODEL_BASE}/ckpts",
            "/runpod-volume/models/hunyuan/ckpts",
            "/runpod-volume/ckpts"
        ]

        for path in possible_paths:
            if os.path.exists(path):
                models_root = path
                print(f"  Found models at: {path}")
                break

    if not models_root or not os.path.exists(models_root):
        return {"error": f"Models not found. Checked: {MODEL_BASE}/ckpts and alternatives. Please ensure models are downloaded."}

    print(f"Running I2V generation (3-second optimized)...")
    print(f"  Prompt: {prompt[:80]}...")
    print(f"  Resolution: {resolution}, Frames: {num_frames} ({num_frames/fps:.1f}s @ {fps}fps)")
    print(f"  Steps: {infer_steps}, Flow shift: {flow_shift}")
    print(f"  Models root: {models_root}")

    # List contents of models_root for debugging
    try:
        contents = list(Path(models_root).iterdir())
        print(f"  Models root contents: {[c.name for c in contents[:10]]}")
    except Exception as e:
        print(f"  Could not list models_root: {e}")

    # HunyuanVideo-I2V expects MODEL_BASE env var to point to ckpts folder containing:
    #   - hunyuan-video-i2v-720p/transformers/mp_rank_00_model_states.pt
    #   - text_encoder_i2v/ (LLaVA)
    #   - text_encoder_2/ (CLIP)
    # models_root from setup_i2v_models() is MODEL_BASE/ckpts
    model_base = models_root  # /runpod-volume/models/hunyuan/ckpts

    # Build absolute path for I2V model weights
    i2v_dit_weight = f"{model_base}/hunyuan-video-i2v-720p/transformers/mp_rank_00_model_states.pt"

    # Get cfg_scale from kwargs
    cfg_scale = kwargs.get("cfg_scale", 6.0)

    cmd = [
        "python3", "/app/HunyuanVideo-I2V/sample_image2video.py",
        "--model-base", model_base,
        "--i2v-dit-weight", i2v_dit_weight,  # Absolute path to avoid relative path issues
        "--prompt", prompt,
        "--i2v-mode",
        "--i2v-image-path", str(image_path),
        "--i2v-resolution", resolution,
        "--video-length", str(num_frames),
        "--flow-shift", str(flow_shift),
        "--infer-steps", str(infer_steps),
        "--embedded-cfg-scale", str(cfg_scale),  # Required for guidance distilled model
        "--seed", str(seed),
        "--use-cpu-offload",
        "--save-path", str(OUTPUT_DIR)
    ]

    if stability:
        cmd.append("--i2v-stability")

    env = os.environ.copy()
    env["PYTHONPATH"] = "/app/HunyuanVideo-I2V"
    env["MODEL_BASE"] = model_base  # Script's constants.py uses this for text encoder paths

    # Working directory should be the script directory
    work_dir = "/app/HunyuanVideo-I2V"

    print(f"  Command: python3 sample_image2video.py ...")
    print(f"  Working dir: {work_dir}")

    start = time.time()
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env=env,
        cwd=work_dir,
        timeout=600  # 10 min timeout for 3s video
    )
    elapsed = time.time() - start

    print(f"  Completed in {elapsed:.1f}s")

    if result.returncode != 0:
        print(f"STDOUT: {result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout}")
        print(f"STDERR: {result.stderr[-2000:] if len(result.stderr) > 2000 else result.stderr}")
        return {"error": f"I2V generation failed: {result.stderr[-500:]}"}

    # Find output video
    output_videos = list(OUTPUT_DIR.glob("*.mp4"))
    if not output_videos:
        # Also check working directory
        output_videos = list(Path(work_dir).glob("**/*.mp4"))

    if not output_videos:
        return {"error": "No output video generated"}

    output_video = sorted(output_videos, key=lambda x: x.stat().st_mtime)[-1]  # Get latest
    print(f"  Output video: {output_video}")

    with open(output_video, 'rb') as f:
        video_base64 = base64.b64encode(f.read()).decode('utf-8')

    # Get actual video duration
    try:
        probe_cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                     "-of", "default=noprint_wrappers=1:nokey=1", str(output_video)]
        probe_result = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=10)
        video_duration = float(probe_result.stdout.strip()) if probe_result.returncode == 0 else num_frames/fps
    except:
        video_duration = num_frames / fps

    return {
        "video_base64": video_base64,
        "duration": video_duration,
        "generation_time": elapsed,
        "num_frames": num_frames,
        "resolution": resolution,
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

    # Model path
    models_root = f"{MODEL_BASE}/hunyuan-avatar/ckpts"

    cmd = [
        "python3", "/app/HunyuanVideo/sample_video.py",
        "--model-base", models_root,
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
        # Check disk space before starting (skip for cleanup and disk_check modes)
        disk = get_disk_usage()
        if disk:
            print(f"  Disk: {disk['free_gb']:.1f}GB free / {disk['total_gb']:.1f}GB total ({disk['used_percent']:.1f}% used)")
            # Only block if low disk AND not a maintenance mode
            if disk['free_gb'] < 5 and mode not in ["cleanup", "disk_check"]:
                return {"error": f"Low disk space: only {disk['free_gb']:.1f}GB free. Please clear the volume."}

        # Download models if needed (skip for maintenance modes)
        if mode not in ["cleanup", "disk_check"]:
            download_models_if_needed(mode)

        # Clean temp directory
        cleanup_temp_files()

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
            # Supports both URL and base64 input
            image_url = job_input.get("image_url")
            image_base64 = job_input.get("image_base64")
            prompt = job_input.get("prompt", "")

            image_path = TEMP_DIR / "input_image.jpg"
            output_path = OUTPUT_DIR / "i2v_output.mp4"

            # Handle image input (URL or base64)
            if image_base64:
                print("  Using base64 image input for I2V")
                image_data = base64.b64decode(image_base64)
                with open(image_path, 'wb') as f:
                    f.write(image_data)
                print(f"  Image saved: {len(image_data)//1024}KB")
            elif image_url:
                if not download_file(image_url, image_path):
                    return {"error": "Failed to download image"}
            else:
                return {"error": "I2V mode requires image_url or image_base64"}

            result = generate_i2v(
                image_path, prompt, output_path,
                resolution=job_input.get("resolution", "540p"),
                num_frames=job_input.get("num_frames", 77),  # (77-1)=76 divisible by 4
                stability=job_input.get("stability", True),
                flow_shift=job_input.get("flow_shift", 7.0),
                infer_steps=job_input.get("infer_steps", 30),  # 30 for speed
                cfg_scale=job_input.get("cfg_scale", 6.0),  # Required for guidance distilled model
                seed=job_input.get("seed", 42),
                fps=job_input.get("fps", 25)
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

        elif mode == "disk_check":
            # Disk check mode - returns disk usage info
            disk = get_disk_usage()

            # Also check specific directories
            dir_sizes = {}
            check_dirs = [
                "/runpod-volume",
                str(HF_CACHE_DIR),
                "/app",
                "/tmp"
            ]

            for d in check_dirs:
                try:
                    result = subprocess.run(
                        ["du", "-sh", d],
                        capture_output=True,
                        text=True,
                        timeout=30
                    )
                    if result.returncode == 0:
                        size = result.stdout.strip().split()[0]
                        dir_sizes[d] = size
                except:
                    dir_sizes[d] = "unknown"

            return {
                "success": True,
                "disk": disk,
                "directories": dir_sizes,
                "message": "Disk check completed"
            }

        elif mode == "cleanup":
            # Force cleanup mode - clears cache and temp files
            disk_before = get_disk_usage()
            cleaned = []

            # Clean temp directories
            cleanup_temp_files()
            cleaned.append("temp files")

            # Clean HuggingFace cache (optional, controlled by parameter)
            if job_input.get("clear_hf_cache", False):
                try:
                    if HF_CACHE_DIR.exists():
                        shutil.rmtree(HF_CACHE_DIR)
                        HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
                        cleaned.append("HuggingFace cache")
                except Exception as e:
                    print(f"  HF cache cleanup error: {e}")

            # Clean /tmp
            try:
                for f in Path("/tmp").glob("*"):
                    if f.is_file():
                        f.unlink()
                    elif f.is_dir() and f.name.startswith("tmp"):
                        shutil.rmtree(f)
                cleaned.append("/tmp files")
            except Exception as e:
                print(f"  /tmp cleanup error: {e}")

            disk_after = get_disk_usage()
            freed = disk_after['free_gb'] - disk_before['free_gb'] if disk_before and disk_after else 0

            return {
                "success": True,
                "disk_before": disk_before,
                "disk_after": disk_after,
                "freed_gb": round(freed, 2),
                "cleaned": cleaned,
                "message": f"Cleanup completed, freed {freed:.2f}GB"
            }

        else:
            return {"error": f"Unknown mode: {mode}. Use 'avatar', 'i2v', 't2v', 'disk_check', or 'cleanup'"}

        # Post-job cleanup
        cleanup_temp_files()

        # Check disk space after job
        disk = get_disk_usage()
        if disk:
            print(f"  Disk after: {disk['free_gb']:.1f}GB free ({disk['used_percent']:.1f}% used)")

        print("=" * 60)
        print(f"Job completed: {job_id}")
        print("=" * 60 + "\n")

        return result

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

        # Cleanup on error
        cleanup_temp_files()

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
