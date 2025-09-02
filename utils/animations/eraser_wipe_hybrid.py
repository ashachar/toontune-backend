#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Eraser wipe: nearest-neighbor (Voronoi) timing with a one-time CPU precompute,
then a single ffmpeg command for compositing.

- Generates tmap16.png (16-bit grayscale) where each pixel stores t* in [0..1]
  at which the eraser is closest to that pixel (true nearest seed along the path).
- Runs a single ffmpeg command that:
    * loops the time-map,
    * builds a smooth time-varying alpha,
    * preserves foreground color via alphamerge,
    * overlays the eraser PNG along the exact same path.

Requirements:
    pip install numpy opencv-python
    ffmpeg (7.0+ recommended) available in PATH.

Tested on: macOS (Apple Silicon) with only VideoToolbox; no OpenCL/CUDA needed.
"""

import math
import os
import subprocess
import tempfile
from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np

# ---------------------------
# Path generation (constant-speed)
# ---------------------------

@dataclass
class PathParams:
    pattern: str = "s_curve"  # "s_curve" | "ellipse" | "vertical_sweep" | "figure8"
    center_x: float = None     # if None -> width/2
    center_y: float = None     # if None -> height/2
    radius_x: float = None     # if None -> width*0.35
    radius_y: float = None     # if None -> height*0.35
    cycles: float = 2.5        # for s_curve
    figure8_aspect: float = 0.75  # for figure8


def _parametric_point(pattern: str, u: float, W: int, H: int,
                      cx: float, cy: float, rx: float, ry: float,
                      cycles: float, figure8_aspect: float) -> Tuple[float, float]:
    """
    Return (x, y) for u in [0,1] along the chosen path.
    The parameter u is NOT constant-speed; we will re-sample by arc length later.
    """
    if pattern == "s_curve":
        # Vertical sweep with horizontal sine oscillations
        y = H * (0.05 + 0.90 * u)
        x = cx + rx * 0.7 * math.sin(cycles * 2 * math.pi * u)
        return x, y
    elif pattern == "ellipse":
        ang = 2 * math.pi * u
        return cx + rx * math.cos(ang), cy + ry * math.sin(ang)
    elif pattern == "vertical_sweep":
        return cx, H * (0.05 + 0.90 * u)
    elif pattern == "figure8":
        # Lissajous-like "∞"
        ang = 2 * math.pi * u
        x = cx + rx * math.sin(ang)
        y = cy + ry * math.sin(ang) * math.cos(ang) * figure8_aspect
        return x, y
    else:
        # default: vertical sweep
        return cx, H * (0.05 + 0.90 * u)


def _resample_constant_speed(W: int, H: int, params: PathParams,
                             samples: int = 2048) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample the path densely, compute arc length, then re-sample by equal arc-length
    to obtain constant-speed positions and cumulative length fractions in [0,1].

    Returns:
        P:  (M, 2) float32 positions [[x,y], ...]
        S:  (M,)   float32 normalized arc length (0..1)
    """
    cx = params.center_x if params.center_x is not None else W / 2.0
    cy = params.center_y if params.center_y is not None else H / 2.0
    rx = params.radius_x if params.radius_x is not None else W * 0.35
    ry = params.radius_y if params.radius_y is not None else H * 0.35

    # Dense parameter sampling (not constant-speed yet)
    u = np.linspace(0.0, 1.0, samples, dtype=np.float32)
    pts = np.array([_parametric_point(params.pattern, float(ui), W, H, cx, cy, rx, ry,
                                      params.cycles, params.figure8_aspect)
                    for ui in u], dtype=np.float32)

    seg = pts[1:] - pts[:-1]
    seg_len = np.hypot(seg[:, 0], seg[:, 1])
    cum = np.concatenate([[0.0], np.cumsum(seg_len, dtype=np.float64)])
    total = float(cum[-1]) if cum[-1] > 0 else 1.0
    s_norm = (cum / total).astype(np.float32)   # 0..1 along arc length

    # Re-sample by equal arc-length (constant-speed)
    M = max(256, min(samples, 4096))
    s_eq = np.linspace(0.0, 1.0, M, dtype=np.float32)
    # Interpolate x,y by arc-length
    x = np.interp(s_eq, s_norm, pts[:, 0])
    y = np.interp(s_eq, s_norm, pts[:, 1])

    P = np.stack([x, y], axis=1).astype(np.float32)  # shape (M,2)
    S = s_eq                                         # shape (M,)
    return P, S


# ---------------------------
# Time-map generation with OpenCV Distance Transform + Labels
# ---------------------------

def _make_seed_coords(P: np.ndarray, S: np.ndarray, step_px: float = 2.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Choose seed points along the constant-speed polyline P spaced about step_px apart.
    Ensures seeds are at least ~2 pixels apart in Euclidean space to avoid 8-connectivity.
    Returns:
        seed_xy: float32 (K,2) seed coordinates
        seed_t:  float32 (K,)   normalized time for each seed (0..1)
    """
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

    seed_xy = P[keep]         # (K,2)
    seed_t  = S[keep]         # (K,)
    # Enforce minimum 2-pixel separation after rounding
    out_xy = [seed_xy[0]]
    out_t  = [seed_t[0]]
    for j in range(1, len(seed_xy)):
        if np.hypot(*(seed_xy[j] - out_xy[-1])) >= 2.0:
            out_xy.append(seed_xy[j])
            out_t.append(seed_t[j])
    return np.array(out_xy, dtype=np.float32), np.array(out_t, dtype=np.float32)


def _label_for_seed(labels: np.ndarray, y: int, x: int) -> int:
    """
    Find the non-zero label around a seed at (y,x) by scanning a small radius.
    DistanceTransformWithLabels sets label 0 on zero-pixels themselves.
    """
    H, W = labels.shape
    for radius in (1, 2, 3, 4):
        y0 = max(0, y - radius); y1 = min(H, y + radius + 1)
        x0 = max(0, x - radius); x1 = min(W, x + radius + 1)
        block = labels[y0:y1, x0:x1]
        nz = block[block > 0]
        if nz.size:
            return int(nz.flat[0])
    return 0


def generate_time_map_png(width: int, height: int,
                          params: PathParams,
                          wipe_start: float = 0.10,
                          wipe_duration: float = 0.90,
                          seed_step_px: float = 2.0,
                          out_png_path: str = "tmap16.png") -> str:
    """
    Create a single-frame 16-bit grayscale PNG where each pixel stores t* in [0,1]:
    the time (normalized to wipe_start..wipe_end) when the eraser is closest.

    Output is saved to out_png_path (size WxH), and the path is returned.
    """
    # 1) Constant-speed path samples
    P, S = _resample_constant_speed(width, height, params, samples=2048)

    # 2) Seeds roughly every ~2 px along the path
    seed_xy, seed_t = _make_seed_coords(P, S, step_px=seed_step_px)

    # 3) Build seed image: 0 at seed pixels, 255 elsewhere
    img = np.full((height, width), 255, dtype=np.uint8)
    seed_px = np.round(seed_xy).astype(np.int32)
    seed_px[:, 0] = np.clip(seed_px[:, 0], 0, width - 1)
    seed_px[:, 1] = np.clip(seed_px[:, 1], 0, height - 1)
    for (x, y) in seed_px:
        img[y, x] = 0

    # 4) Distance transform WITH labels (nearest zero pixel label)
    #    DIST_LABEL_PIXEL gives a unique ID per zero-pixel.
    dist, labels = cv2.distanceTransformWithLabels(
        img, distanceType=cv2.DIST_L2, maskSize=cv2.DIST_MASK_3,
        labelType=cv2.DIST_LABEL_PIXEL
    )
    # labels: int32, same HxW

    # 5) Map each seed -> its label, then build LUT(label -> t_norm)
    #    Seeds themselves (zero pixels) have label 0; find label from neighbors.
    lut_size = int(labels.max()) + 1
    lut = np.full(lut_size, -1.0, dtype=np.float32)

    # Build mapping using neighbor labels
    for idx, (x, y) in enumerate(seed_px):
        L = _label_for_seed(labels, int(y), int(x))
        if L > 0:
            lut[L] = seed_t[idx]

    # 6) Convert labels -> t_norm; fill seeds (label==0) directly from seed_t
    tmap = np.empty_like(dist, dtype=np.float32)
    # For normal pixels (label>0), map via LUT
    mask_pos = labels > 0
    tmap[mask_pos] = lut[labels[mask_pos]]
    # For seeds (label==0), write their own t_norm
    tmap[~mask_pos] = 0.0  # temporary
    for idx, (x, y) in enumerate(seed_px):
        tmap[y, x] = seed_t[idx]

    # Safety: any remaining -1 due to rare corner cases → nearest non-neg via blur pass
    missing = tmap < 0
    if np.any(missing):
        # simple 3x3 median fill
        tmap = cv2.medianBlur(np.where(missing, np.nan, tmap).astype(np.float32), 3)
        tmap = np.nan_to_num(tmap, copy=False, nan=0.0)

    # 7) Normalize to 16-bit [0..65535]; NOTE: tmap already 0..1 along wipe
    tmap16 = np.clip(np.round(tmap * 65535.0), 0, 65535).astype(np.uint16)
    cv2.imwrite(out_png_path, tmap16)

    return out_png_path


# ---------------------------
# FFmpeg invocation
# ---------------------------

def _ffprobe_dims_and_fps(path: str) -> Tuple[int, int, float]:
    """Return (width, height, fps) using ffprobe."""
    import json
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,avg_frame_rate",
        "-of", "json", path
    ]
    out = subprocess.check_output(cmd).decode("utf-8")
    info = json.loads(out)["streams"][0]
    w, h = int(info["width"]), int(info["height"])
    num, den = map(int, info["avg_frame_rate"].split("/"))
    fps = (num / den) if den else 25.0
    return w, h, fps


def _overlay_xy_expr(params: PathParams, wipe_start: float, wipe_duration: float,
                     W: int, H: int, eraser_w: int, eraser_h: int) -> Tuple[str, str]:
    # progress: clip((t - wipe_start)/wipe_duration, 0, 1)
    prog = f"clip((t-{wipe_start})/{wipe_duration},0,1)"
    cx = params.center_x if params.center_x is not None else W / 2.0
    cy = params.center_y if params.center_y is not None else H / 2.0
    rx = params.radius_x if params.radius_x is not None else W * 0.35
    ry = params.radius_y if params.radius_y is not None else H * 0.35

    if params.pattern == "s_curve":
        x = f"{cx} + {rx*0.7}*sin(2*PI*{params.cycles}*{prog}) - {eraser_w}/2"
        y = f"{H*0.05} + {H*0.90}*{prog} - {eraser_h}/2"
    elif params.pattern == "ellipse":
        x = f"{cx} + {rx}*cos(2*PI*{prog}) - {eraser_w}/2"
        y = f"{cy} + {ry}*sin(2*PI*{prog}) - {eraser_h}/2"
    elif params.pattern == "figure8":
        x = f"{cx} + {rx}*sin(2*PI*{prog}) - {eraser_w}/2"
        y = f"{cy} + {ry*params.figure8_aspect}*sin(2*PI*{prog})*cos(2*PI*{prog}) - {eraser_h}/2"
    else:  # vertical_sweep
        x = f"{cx} - {eraser_w}/2"
        y = f"{H*0.05} + {H*0.90}*{prog} - {eraser_h}/2"
    return x, y


def create_eraser_wipe_hybrid(
    character_video: str,
    original_video: str,
    eraser_image: str,
    output_video: str,
    wipe_start: float = 0.10,
    wipe_duration: float = 0.90,
    pattern: str = "s_curve",
    seed_step_px: float = 2.0,
    eraser_scale: float = 0.12,   # relative to min(W,H)
    x264_preset: str = "medium",
    crf: int = 18,
) -> None:
    """
    Hybrid pipeline:
      1) Precompute tmap16.png (Voronoi time map) via CPU once.
      2) Single ffmpeg command for the whole effect.

    This preserves color (alphamerge) and synchronizes eraser PNG with the same path.
    """
    # Probe input dims/fps to set consistent canvas
    w0, h0, f0 = _ffprobe_dims_and_fps(character_video)
    w1, h1, f1 = _ffprobe_dims_and_fps(original_video)
    W, H = max(w0, w1), max(h0, h1)
    fps = int(round(max(f0, f1) or 25.0))

    # Prepare path params
    params = PathParams(pattern=pattern)

    # Compute eraser visual size
    base = min(W, H)
    er_w = int(round(base * eraser_scale))
    er_h = er_w  # square

    # Precompute time map in a temp dir
    with tempfile.TemporaryDirectory() as tdir:
        tmap_png = os.path.join(tdir, "tmap16.png")
        generate_time_map_png(
            width=W, height=H,
            params=params,
            wipe_start=wipe_start, wipe_duration=wipe_duration,
            seed_step_px=seed_step_px,
            out_png_path=tmap_png
        )

        # Overlay expressions (match the same path shape, constant-speed)
        ox, oy = _overlay_xy_expr(params, wipe_start, wipe_duration, W, H, er_w, er_h)

        # A little softness (~1 frame) for professional look
        ease = 1.2 / fps

        # Build filter_complex
        # [3:v] is tmap16.png (looped); p(X,Y) will be 0..65535 since format=gray16le
        progress = f"clip((T-{wipe_start})/{wipe_duration},0,1)"
        # smooth threshold around progress - tmap, rolled into 0..255
        mask_expr = (
            f"255*clip(({progress} - p(X,Y)/65535.0)/{ease} + 0.5, 0, 1)"
        )

        filter_parts = [
            # Loop tmap and keep as gray16le
            f"[3:v]fps={fps},format=gray16le,loop=loop=1000000:size=1:start=0,setpts=N/{fps}/TB[tmap]",

            # Build erase mask for current frame time
            f"[tmap]geq=lum='{mask_expr}',format=gray[mask]",
            f"[mask]lut=y=negval[inv_mask]",

            # Prepare inputs
            f"[0:v]fps={fps},scale={W}:{H}:flags=bicubic,format=gbrp[char_rgb]",
            f"[1:v]fps={fps},scale={W}:{H}:flags=bicubic,format=rgba[bg_rgba]",

            # Color-preserving composite
            f"[char_rgb][inv_mask]alphamerge,format=rgba[char_rgba]",
            f"[bg_rgba][char_rgba]overlay=shortest=0:eof_action=pass:format=auto[reveal]",

            # Eraser PNG motion (same path & timing)
            f"[2:v]loop=loop=999999:size=1:start=0,format=rgba,scale={er_w}:{er_h}[eraser]",
            (
              f"[reveal][eraser]overlay="
              f"x='{ox}':y='{oy}':eval=frame:"
              f"enable='between(t,{wipe_start},{wipe_start + wipe_duration})'[outv]"
            ),
        ]
        filt = ";".join(filter_parts)

        cmd = [
            "ffmpeg", "-y", "-hide_banner",
            "-i", character_video,
            "-i", original_video,
            "-loop", "1", "-i", eraser_image,
            "-loop", "1", "-i", tmap_png,
            "-filter_complex", filt,
            "-map", "[outv]", "-map", "1:a?",
            "-c:v", "libx264", "-preset", x264_preset, "-crf", str(crf),
            "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            output_video
        ]
        print("Running ffmpeg:\n", " ".join(cmd))
        subprocess.run(cmd, check=True)
        print(f"✓ Created: {output_video}")


# ---------------------------
# Example run
# ---------------------------

if __name__ == "__main__":
    create_eraser_wipe_hybrid(
        character_video="outputs/runway_scaled_cropped.mp4",
        original_video="uploads/assets/runway_experiment/runway_demo_input.mp4",
        eraser_image="uploads/assets/images/eraser.png",
        output_video="outputs/final_eraser_hybrid.mp4",
        wipe_start=0.10,
        wipe_duration=0.90,
        pattern="s_curve",         # "s_curve", "ellipse", "vertical_sweep", "figure8"
        seed_step_px=2.0,          # seeds ~ every 2px along path → very accurate timing
        eraser_scale=0.12,         # visual size of eraser
        x264_preset="medium",
        crf=18
    )