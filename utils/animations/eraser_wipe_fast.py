#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fast eraser wipe using a precomputed Voronoi time-map:
- Avoids geq over the full frame by using lut2 (preferred) or threshold (fallback).
- Works on macOS with stock FFmpeg (no CUDA/OpenCL required).
- Preserves color via alphamerge and keeps eraser PNG in sync with the path.

Dependencies:
  pip install numpy opencv-python
  ffmpeg 7.0+ available in PATH
"""

import math
import os
import shlex
import subprocess
import tempfile
from dataclasses import dataclass
from typing import List, Tuple, Optional, Literal

import cv2
import numpy as np


# ---------------------------
# Path and time-map generation (OpenCV), same as before
# ---------------------------

@dataclass
class PathParams:
    pattern: str = "s_curve"   # "s_curve" | "ellipse" | "vertical_sweep" | "figure8"
    center_x: float = None
    center_y: float = None
    radius_x: float = None
    radius_y: float = None
    cycles: float = 2.5
    figure8_aspect: float = 0.75


def _parametric_point(pattern: str, u: float, W: int, H: int,
                      cx: float, cy: float, rx: float, ry: float,
                      cycles: float, figure8_aspect: float) -> Tuple[float, float]:
    if pattern == "s_curve":
        y = H * (0.05 + 0.90 * u)
        x = cx + rx * 0.7 * math.sin(cycles * 2 * math.pi * u)
        return x, y
    elif pattern == "ellipse":
        ang = 2 * math.pi * u
        return cx + rx * math.cos(ang), cy + ry * math.sin(ang)
    elif pattern == "vertical_sweep":
        return cx, H * (0.05 + 0.90 * u)
    elif pattern == "figure8":
        ang = 2 * math.pi * u
        x = cx + rx * math.sin(ang)
        y = cy + ry * math.sin(ang) * math.cos(ang) * figure8_aspect
        return x, y
    else:
        return cx, H * (0.05 + 0.90 * u)


def _resample_constant_speed(W: int, H: int, params: PathParams,
                             samples: int = 2048) -> Tuple[np.ndarray, np.ndarray]:
    cx = params.center_x if params.center_x is not None else W / 2.0
    cy = params.center_y if params.center_y is not None else H / 2.0
    rx = params.radius_x if params.radius_x is not None else W * 0.35
    ry = params.radius_y if params.radius_y is not None else H * 0.35

    u = np.linspace(0.0, 1.0, samples, dtype=np.float32)
    pts = np.array([_parametric_point(params.pattern, float(ui), W, H, cx, cy, rx, ry,
                                      params.cycles, params.figure8_aspect)
                    for ui in u], dtype=np.float32)
    seg = pts[1:] - pts[:-1]
    seg_len = np.hypot(seg[:, 0], seg[:, 1])
    cum = np.concatenate([[0.0], np.cumsum(seg_len, dtype=np.float64)])
    total = float(cum[-1]) if cum[-1] > 0 else 1.0
    s_norm = (cum / total).astype(np.float32)

    M = max(256, min(samples, 4096))
    s_eq = np.linspace(0.0, 1.0, M, dtype=np.float32)
    x = np.interp(s_eq, s_norm, pts[:, 0])
    y = np.interp(s_eq, s_norm, pts[:, 1])
    P = np.stack([x, y], axis=1).astype(np.float32)
    S = s_eq
    return P, S


def _make_seed_coords(P: np.ndarray, S: np.ndarray, step_px: float = 2.0) -> Tuple[np.ndarray, np.ndarray]:
    if len(P) < 2:
        return P.copy(), S.copy()
    keep = [0]
    last = P[0]
    for i in range(1, len(P)):
        if np.hypot(*(P[i] - last)) >= step_px:
            keep.append(i)
            last = P[i]
    if keep[-1] != len(P) - 1:
        keep.append(len(P) - 1)
    seed_xy = P[keep]
    seed_t = S[keep]
    out_xy = [seed_xy[0]]
    out_t = [seed_t[0]]
    for j in range(1, len(seed_xy)):
        if np.hypot(*(seed_xy[j] - out_xy[-1])) >= 2.0:
            out_xy.append(seed_xy[j]); out_t.append(seed_t[j])
    return np.array(out_xy, dtype=np.float32), np.array(out_t, dtype=np.float32)


def _label_for_seed(labels: np.ndarray, y: int, x: int) -> int:
    H, W = labels.shape
    for r in (1, 2, 3, 4):
        y0 = max(0, y - r); y1 = min(H, y + r + 1)
        x0 = max(0, x - r); x1 = min(W, x + r + 1)
        block = labels[y0:y1, x0:x1]
        nz = block[block > 0]
        if nz.size:
            return int(nz.flat[0])
    return 0


def generate_time_map_png(width: int, height: int, params: PathParams,
                          seed_step_px: float,
                          out_png_path: str) -> str:
    """Write 16‑bit grayscale PNG time‑map: pixel stores t* in [0..1]."""
    P, S = _resample_constant_speed(width, height, params, samples=2048)
    seed_xy, seed_t = _make_seed_coords(P, S, step_px=seed_step_px)

    img = np.full((height, width), 255, dtype=np.uint8)
    seed_px = np.round(seed_xy).astype(np.int32)
    seed_px[:, 0] = np.clip(seed_px[:, 0], 0, width - 1)
    seed_px[:, 1] = np.clip(seed_px[:, 1], 0, height - 1)
    for (x, y) in seed_px:
        img[y, x] = 0

    dist, labels = cv2.distanceTransformWithLabels(
        img, distanceType=cv2.DIST_L2, maskSize=cv2.DIST_MASK_3,
        labelType=cv2.DIST_LABEL_PIXEL
    )
    lut_size = int(labels.max()) + 1
    lut = np.full(lut_size, -1.0, dtype=np.float32)
    for idx, (x, y) in enumerate(seed_px):
        L = _label_for_seed(labels, int(y), int(x))
        if L > 0:
            lut[L] = seed_t[idx]

    tmap = np.empty_like(dist, dtype=np.float32)
    mask_pos = labels > 0
    tmap[mask_pos] = lut[labels[mask_pos]]
    for idx, (x, y) in enumerate(seed_px):
        tmap[y, x] = seed_t[idx]
    
    # Save as 8-bit for lut2 efficiency (256x256 LUT instead of 65536x65536)
    tmap8 = np.clip(np.round(tmap * 255.0), 0, 255).astype(np.uint8)
    cv2.imwrite(out_png_path, tmap8)
    return out_png_path


# ---------------------------
# FFmpeg helpers
# ---------------------------

def _ffprobe_dims_fps(path: str) -> Tuple[int, int, float]:
    import json
    out = subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,avg_frame_rate",
        "-of", "json", path
    ]).decode("utf-8")
    st = json.loads(out)["streams"][0]
    w, h = int(st["width"]), int(st["height"])
    num, den = map(int, st["avg_frame_rate"].split("/"))
    fps = (num / den) if den else 25.0
    return w, h, fps


def _filter_exists(name: str) -> bool:
    try:
        out = subprocess.check_output(["ffmpeg", "-hide_banner", "-filters"], 
                                     stderr=subprocess.STDOUT).decode("utf-8")
    except Exception:
        return False
    return any((" " + name + " " in line) or line.strip().startswith(name) 
              for line in out.splitlines())


def _overlay_xy_expr(params: PathParams, wipe_start: float, wipe_duration: float,
                     W: int, H: int,
                     tip_x_ratio: float, tip_y_ratio: float) -> Tuple[str, str]:
    """
    Return overlay x,y expressions that place the OVERLAY'S TIP on the path.
    We subtract overlay_w*tip_x_ratio and overlay_h*tip_y_ratio to anchor the tip pixel.
    """
    prog = f"clip((t-{wipe_start})/{wipe_duration},0,1)"
    cx = params.center_x if params.center_x is not None else W / 2.0
    cy = params.center_y if params.center_y is not None else H / 2.0
    rx = params.radius_x if params.radius_x is not None else W * 0.35
    ry = params.radius_y if params.radius_y is not None else H * 0.35

    # Expressions for the tip position (path) in output coords:
    if params.pattern == "s_curve":
        tip_x = f"{cx} + {rx*0.7}*sin(2*PI*{params.cycles}*{prog})"
        tip_y = f"{H*0.05} + {H*0.90}*{prog}"
    elif params.pattern == "ellipse":
        tip_x = f"{cx} + {rx}*cos(2*PI*{prog})"
        tip_y = f"{cy} + {ry}*sin(2*PI*{prog})"
    elif params.pattern == "figure8":
        tip_x = f"{cx} + {rx}*sin(2*PI*{prog})"
        tip_y = f"{cy} + {ry*params.figure8_aspect}*sin(2*PI*{prog})*cos(2*PI*{prog})"
    else:  # vertical_sweep
        tip_x = f"{cx}"
        tip_y = f"{H*0.05} + {H*0.90}*{prog}"

    # Convert tip coords into overlay's top-left by subtracting tip offsets in overlay space
    ox = f"{tip_x} - overlay_w*{tip_x_ratio}"
    oy = f"{tip_y} - overlay_h*{tip_y_ratio}"
    return ox, oy


# --- New: compute path vertical bounds to drive auto sizing ---

def _path_minmax_y(W: int, H: int, params: PathParams) -> Tuple[float, float]:
    """Sample the path at high resolution to get (min_y, max_y)."""
    P, _ = _resample_constant_speed(W, H, params, samples=4096)
    ys = P[:, 1]
    return float(np.min(ys)), float(np.max(ys))


def _required_height_for_offscreen(H: int, min_y: float, tip_y_ratio: float,
                                   margin_px: int) -> int:
    """
    Compute minimal overlay height so when the tip is at min_y, the overlay bottom
    is at least margin_px below the frame bottom:
        min_y - height*tip_y_ratio + height >= H + margin_px
      => height * (1 - tip_y_ratio) >= H + margin_px - min_y
      => height >= (H + margin_px - min_y) / (1 - tip_y_ratio)
    """
    denom = max(1e-6, (1.0 - tip_y_ratio))
    needed = (H + float(margin_px) - float(min_y)) / denom
    return int(math.ceil(max(1.0, needed)))


# ---------------------------
# Main entry: fast compositing with lut2 (preferred) or threshold (fallback)
# ---------------------------

def create_eraser_wipe_fast(
    character_video: str,
    original_video: str,
    eraser_image: str,
    output_video: str,
    wipe_start: float = 0.10,
    wipe_duration: float = 0.90,
    pattern: str = "s_curve",
    seed_step_px: float = 2.0,
    
    # === NEW SIZE OPTIONS ===
    eraser_size_mode: Literal["auto_offscreen", "ratio", "px"] = "auto_offscreen",
    eraser_height_ratio: float = 1.10,          # used when mode == "ratio"
    eraser_height_px: Optional[int] = None,     # used when mode == "px"
    offscreen_margin_px: int = 32,              # extra pixels the hand should extend below the frame in auto mode
    
    # Tip anchor in PNG (keep these matching your art)
    tip_x_ratio: float = 0.50,                  # horizontally centered
    tip_y_ratio: float = 0.12,                  # tip 12% from top
    
    # encoding
    crf: int = 18,
    x264_preset: str = "medium",
) -> None:
    # Probe I/O
    w0, h0, f0 = _ffprobe_dims_fps(character_video)
    w1, h1, f1 = _ffprobe_dims_fps(original_video)
    W, H = max(w0, w1), max(h0, h1)
    fps = int(round(max(f0, f1) or 25.0))
    
    # Get video duration to calculate proper loop count
    def _get_video_duration(path: str) -> float:
        import json
        out = subprocess.check_output([
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "json", path
        ]).decode("utf-8")
        return float(json.loads(out)["format"]["duration"])
    
    # Calculate proper loop count based on video duration
    video_duration = max(_get_video_duration(character_video), _get_video_duration(original_video))
    total_frames = int(fps * video_duration) + 10  # Add small buffer
    loop_count = total_frames
    print(f"Video duration: {video_duration:.2f}s, FPS: {fps}, Total frames: {total_frames}, Loop count: {loop_count}")

    # Path params
    params = PathParams(pattern=pattern)

    # Build time‑map once (8‑bit for lut2 efficiency)
    with tempfile.TemporaryDirectory() as tdir:
        tmap_png = os.path.join(tdir, "tmap8.png")
        print(f"Generating Voronoi time map at {W}x{H}...")
        generate_time_map_png(W, H, params, seed_step_px, tmap_png)
        print(f"Time map saved: {tmap_png}")

        # --- Decide eraser target height ---
        if eraser_size_mode == "px":
            target_h = int(max(1, eraser_height_px or 1))
        elif eraser_size_mode == "ratio":
            target_h = int(round(max(1.0, H * float(eraser_height_ratio))))
        else:  # "auto_offscreen" (default)
            min_y, _ = _path_minmax_y(W, H, params)
            target_h = _required_height_for_offscreen(H, min_y, tip_y_ratio, offscreen_margin_px)
            print(f"Auto sizing: min_y={min_y:.1f}, required height={target_h}px ({target_h/H:.2f}×H) for {offscreen_margin_px}px margin")

        # Overlay x,y expressions with tip anchoring
        ox, oy = _overlay_xy_expr(params, wipe_start, wipe_duration, W, H,
                                  tip_x_ratio=tip_x_ratio, tip_y_ratio=tip_y_ratio)

        # Decide path: lut2 preferred, threshold fallback
        use_lut2 = _filter_exists("lut2")
        use_threshold = _filter_exists("threshold")

        if not use_lut2 and not use_threshold:
            raise RuntimeError(
                "Neither lut2 nor threshold filter is available in your FFmpeg build. "
                "Please install a standard FFmpeg with libavfilter 'lut2' or 'threshold' support."
            )

        print(f"Using filter: {'lut2' if use_lut2 else 'threshold'}")

        # Build filter_complex
        parts = []

        # Loop tmap (single frame) as 8-bit gray
        parts += [
            f"[3:v]fps={fps},format=gray,loop=loop={loop_count}:size=1:start=0,setpts=N/{fps}/TB[tmap]"
        ]

        # Build per-frame PROGRESS as 1x1 gray via geq (1 pixel only!), then scale to W×H
        parts += [
            f"color=c=gray:s=1x1:r={fps}[p1]",
            f"[p1]geq=lum='255*clip((T-{wipe_start})/{wipe_duration},0,1)'[p2]",
            f"[p2]scale={W}:{H}:flags=neighbor,format=gray[progress]"
        ]

        # Create mask: tmap <= progress ? 255 : 0 (binary, fast)
        if use_lut2:
            # lut2: table-driven per-pixel op (very fast)
            # x=tmap value, y=progress value
            parts += [
                f"[tmap][progress]lut2=c0='if(lte(x,y),255,0)',format=gray[mask]"
            ]
        else:
            # threshold: needs 4 inputs
            parts += [
                f"color=c=black:s={W}x{H}:r={fps}[black]",
                f"color=c=white:s={W}x{H}:r={fps}[white]",
                f"[tmap][progress][black][white]threshold[mask]"
            ]

        # Invert to use as foreground alpha
        parts += [
            f"[mask]lut=y=negval[inv_mask]"
        ]

        # Prepare character & background and composite with preserved color
        parts += [
            f"[0:v]fps={fps},scale={W}:{H}:flags=bicubic,format=gbrp[char_rgb]",
            f"[1:v]fps={fps},scale={W}:{H}:flags=bicubic,format=rgba[bg_rgba]",
            f"[char_rgb][inv_mask]alphamerge,format=rgba[char_rgba]",
            f"[bg_rgba][char_rgba]overlay=shortest=1:format=auto[reveal]"
        ]

        # Eraser scaling (aspect-preserving) and tip-anchored overlay.
        # Use Lanczos for higher-quality upscales if target_h > source_h.
        parts += [
            f"[2:v]loop=loop={loop_count}:size=1:start=0,format=rgba,scale=-1:{target_h}:flags=lanczos[eraser]",
            (
              f"[reveal][eraser]overlay=x='{ox}':y='{oy}':eval=frame:"
              f"enable='between(t,{wipe_start},{wipe_start + wipe_duration})'[with_eraser]"
            ),
            f"[with_eraser]trim=duration={video_duration}[outv]"
        ]

        filter_complex = ";".join(parts)

        cmd = [
            "ffmpeg", "-y", "-hide_banner",
            "-i", character_video,
            "-i", original_video,
            "-loop", "1", "-i", eraser_image,
            "-loop", "1", "-i", tmap_png,
            "-filter_complex", filter_complex,
            "-map", "[outv]", "-map", "1:a?",
            "-c:v", "libx264", "-preset", x264_preset, "-crf", str(crf),
            "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            output_video
        ]
        
        print("Running FFmpeg with optimized filters...")
        print(" ".join(shlex.quote(x) for x in cmd[:5]) + " ...")
        subprocess.run(cmd, check=True)
        print(f"✓ Created: {output_video}")


# ---------------------------
# Example usage
# ---------------------------

if __name__ == "__main__":
    # Default: auto_offscreen ensures the hand ALWAYS extends below the frame,
    # with a small safety margin (offscreen_margin_px).
    create_eraser_wipe_fast(
        character_video="outputs/runway_scaled_cropped.mp4",
        original_video="uploads/assets/runway_experiment/runway_demo_input.mp4",
        eraser_image="uploads/assets/images/eraser.png",
        output_video="outputs/final_eraser_fast.mp4",
        wipe_start=0.10,
        wipe_duration=0.90,
        pattern="s_curve",
        seed_step_px=2.0,
        
        # Sizing options:
        eraser_size_mode="auto_offscreen",   # or "ratio" / "px"
        eraser_height_ratio=1.10,            # used if eraser_size_mode="ratio"
        eraser_height_px=None,               # used if eraser_size_mode="px"
        offscreen_margin_px=32,              # guarantee bottom is at least 32 px below the frame at path top
        
        # Tip anchor derived from your PNG layout
        tip_x_ratio=0.50,
        tip_y_ratio=0.12,
        
        crf=18,
        x264_preset="medium"
    )