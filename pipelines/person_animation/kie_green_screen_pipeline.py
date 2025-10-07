"""Five-step pipeline to animate a transparent PNG via Kie.ai."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from PIL import Image
import numpy as np
import requests
import replicate
import cv2
import shutil

# Ensure the project root is importable when running as a script
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.utils.kie import KieAIClient, upload_image

load_dotenv()

OUTPUT_ROOT = Path("outputs/kie_green_screen_pipeline")
GREEN_COLOR = (0, 255, 0, 255)
CANVAS_SIZE = (1280, 720)
RVM_MODEL_VERSION = "arielreplicate/robust_video_matting:73d2128a371922d5d1abf0712a1d974be0e4e2358cc1218e4e34714767232bac"
DESPILL_STRENGTH = 1.0
DESPILL_RESTORE = 0.25
def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_rgba(image_path: Path) -> Image.Image:
    image = Image.open(image_path)
    if image.mode != "RGBA":
        image = image.convert("RGBA")
    return image

def step_add_green_screen(image_path: Path) -> Path:
    original = load_rgba(image_path)

    canvas = Image.new("RGBA", CANVAS_SIZE, GREEN_COLOR)

    resized = original
    if original.width > CANVAS_SIZE[0] or original.height > CANVAS_SIZE[1]:
        scale = min(CANVAS_SIZE[0] / original.width, CANVAS_SIZE[1] / original.height) * 0.95
        new_size = (int(original.width * scale), int(original.height * scale))
        resized = original.resize(new_size, Image.LANCZOS)

    x = (CANVAS_SIZE[0] - resized.width) // 2
    y = (CANVAS_SIZE[1] - resized.height) // 2
    canvas.paste(resized, (x, y), mask=resized.split()[-1])

    output_path = OUTPUT_ROOT / "step2_green_screen.png"
    canvas.save(output_path)
    return output_path


def step_generate_video(start_frame: Path, prompt: str, *, use_fast: bool = False) -> Path:
    client = KieAIClient()
    frame_url = upload_image(str(start_frame))

    task_id = client.generate_video(
        prompt=prompt,
        image_urls=[frame_url],
        model="veo3_fast" if use_fast else "veo3",
        aspect_ratio="16:9",
        enable_fallback=True,
    )
    result_urls = client.wait_for_completion(task_id)
    if not result_urls:
        raise RuntimeError("Kie.ai returned no result URLs")

    raw_video = OUTPUT_ROOT / "step3_kie_raw.mp4"
    client.download_video(result_urls[0], raw_video)
    return raw_video


def run_ffmpeg(args: list[str], *, description: Optional[str] = None) -> None:
    if description:
        print(description)
    result = subprocess.run(args, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg failed: {result.stderr}")


def step_generate_alpha_mask(video_path: Path) -> Path:
    """Use Robust Video Matting via Replicate to extract an alpha mask."""
    if not os.getenv("REPLICATE_API_TOKEN"):
        raise RuntimeError("REPLICATE_API_TOKEN not found. Export it or add to backend/.env")

    alpha_video_path = OUTPUT_ROOT / "rvm_alpha_mask.mp4"

    print("Running Robust Video Matting (Replicate) to estimate alpha mask…")
    with open(video_path, "rb") as video_file:
        result = replicate.run(
            RVM_MODEL_VERSION,
            input={
                "input_video": video_file,
                "output_type": "alpha-mask",
            },
        )

    mask_url = str(result)
    response = requests.get(mask_url, stream=True, timeout=600)
    response.raise_for_status()

    with open(alpha_video_path, "wb") as handle:
        for chunk in response.iter_content(chunk_size=8192):
            handle.write(chunk)

    print(f"   Alpha mask downloaded -> {alpha_video_path}")
    return alpha_video_path


def step_apply_alpha_mask(foreground_video: Path, alpha_video: Path) -> Path:
    """Merge the predicted alpha mask with the Kie render and apply despill."""
    output_path = OUTPUT_ROOT / "step4_no_bg.webm"
    temp_dir = ensure_dir(OUTPUT_ROOT / "temp_rvm_frames")

    # Clear previous frames
    for existing in temp_dir.glob("*.png"):
        existing.unlink()

    fg_cap = cv2.VideoCapture(str(foreground_video))
    mask_cap = cv2.VideoCapture(str(alpha_video))

    if not fg_cap.isOpened() or not mask_cap.isOpened():
        raise RuntimeError("Failed to open video or alpha mask for processing")

    fps = fg_cap.get(cv2.CAP_PROP_FPS) or 30
    frame_count = 0

    while True:
        ret_fg, fg_frame = fg_cap.read()
        ret_mask, mask_frame = mask_cap.read()
        if not ret_fg or not ret_mask:
            break

        if mask_frame.ndim == 3:
            alpha = mask_frame[:, :, 0]
        else:
            alpha = mask_frame

        alpha = alpha.astype(np.uint8)
        bgr = fg_frame.astype(np.float32)

        # Despill: reduce green relative to red/blue and restore luminance
        b = bgr[:, :, 0]
        g = bgr[:, :, 1]
        r = bgr[:, :, 2]

        max_rb = np.maximum(r, b)
        spill = np.maximum(g - max_rb, 0) * DESPILL_STRENGTH
        g = g - spill
        correction = spill * DESPILL_RESTORE
        r = np.clip(r + correction, 0, 255)
        b = np.clip(b + correction, 0, 255)
        g = np.clip(g, 0, 255)

        bgra = np.stack((b, g, r, alpha), axis=-1).astype(np.uint8)

        frame_path = temp_dir / f"frame_{frame_count:04d}.png"
        cv2.imwrite(str(frame_path), bgra)
        frame_count += 1

    fg_cap.release()
    mask_cap.release()

    if frame_count == 0:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise RuntimeError("No frames processed while applying alpha mask")

    run_ffmpeg(
        [
            "ffmpeg",
            "-y",
            "-framerate",
            str(fps),
            "-i",
            str(temp_dir / "frame_%04d.png"),
            "-c:v",
            "libvpx-vp9",
            "-pix_fmt",
            "yuva420p",
            str(output_path),
        ],
        description="Encoding alpha video from RVM mask…",
    )

    shutil.rmtree(temp_dir, ignore_errors=True)
    return output_path


def build_prompt() -> str:
    return (
        "Generate an 8 second clip of the woman holding a key and tablet. "
        "Keep the camera static, centered on her body. The subject must not speak, mouth words, "
        "or move her lips at any point—only perform subtle upper-body motion. Have her lean "
        "forward a touch, give a gentle confident smile, move the key and tablet slightly, and "
        "arrive back at the exact starting pose right at the end of the 8 seconds so the clip can loop cleanly."
    )


def main(image_path: str, *, use_fast: bool = False, reuse_video: bool = False) -> None:
    source = Path(image_path).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"Image not found: {source}")

    ensure_dir(OUTPUT_ROOT)
    step1_path = OUTPUT_ROOT / "step1_original.png"
    load_rgba(source).save(step1_path)

    print("Step 1: Loaded transparent PNG.")
    green_screen_path = step_add_green_screen(step1_path)
    print(f"Step 2: Added green background -> {green_screen_path}")

    raw_video_path = OUTPUT_ROOT / "step3_kie_raw.mp4"

    if reuse_video:
        if not raw_video_path.exists():
            raise FileNotFoundError(
                f"{raw_video_path} not found. Run without --reuse-video first to generate it."
            )
        print("Step 3: Reusing existing Kie.ai render.")
    else:
        prompt = build_prompt()
        print("Step 3: Submitting to Kie.ai...")
        raw_video_path = step_generate_video(green_screen_path, prompt, use_fast=use_fast)
        print(f"   Downloaded raw video -> {raw_video_path}")

    print("Step 4: Estimating alpha mask with Robust Video Matting...")
    alpha_mask_path = step_generate_alpha_mask(raw_video_path)
    print(f"   Alpha mask saved -> {alpha_mask_path}")

    print("Step 5: Encoding transparent WebM with the new alpha...")
    alpha_video_path = step_apply_alpha_mask(raw_video_path, alpha_mask_path)
    print(f"   Saved alpha video -> {alpha_video_path}")

    print("Pipeline complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Animate a PNG via Kie.ai with greenscreen removal.")
    parser.add_argument("image", help="Path to the transparent PNG to animate")
    parser.add_argument("--fast", action="store_true", help="Use veo3_fast model for quicker results")
    parser.add_argument("--reuse-video", action="store_true",
                        help="Skip Kie.ai generation and reuse the existing step3_kie_raw.mp4")
    args = parser.parse_args()

    main(args.image, use_fast=args.fast, reuse_video=args.reuse_video)
