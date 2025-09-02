# utils/animations/eraser_wipe_fixed.py
# -*- coding: utf-8 -*-

import json
import math
import shlex
import subprocess
from dataclasses import dataclass
from typing import List, Tuple, Optional


@dataclass
class VideoProps:
    width: int
    height: int
    fps: float
    duration: float


def _run(cmd: List[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)


def _ffprobe_props(path: str) -> VideoProps:
    """Probe width/height/fps/duration using ffprobe."""
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,avg_frame_rate:format=duration",
        "-of", "json", path
    ]
    proc = _run(cmd)
    data = json.loads(proc.stdout.decode("utf-8"))
    stream = data["streams"][0]
    width = int(stream["width"])
    height = int(stream["height"])
    # avg_frame_rate like "30000/1001"
    num, den = stream.get("avg_frame_rate", "30/1").split("/")
    fps = float(num) / float(den) if float(den) != 0 else 30.0
    duration = float(data["format"].get("duration", "0") or 0.0)
    return VideoProps(width=width, height=height, fps=fps, duration=duration)


def _fmt(x: float) -> str:
    """Format float for FFmpeg expressions (avoid scientific)."""
    return f"{x:.6f}".rstrip("0").rstrip(".") if isinstance(x, float) else str(x)


def _build_piecewise_linear_expr(points: List[Tuple[float, float, float]], coord_index: int) -> str:
    """
    Build a piecewise linear FFmpeg expression over time t.
    points: list of (t, x, y)
    coord_index: 1 for x, 2 for y
    Returns a string expression like:
      if(between(t,t0,t1), v0 + (v1-v0)*(t-t0)/(t1-t0),
         if(between(t,t1,t2), v1 + (v2-v1)*(t-t1)/(t2-t1),
            ...
            v_last))
    """
    assert len(points) >= 2, "Need at least 2 path points"
    expr = ""
    for i in range(len(points) - 1):
        t0, v0 = points[i][0], points[i][coord_index]
        t1, v1 = points[i + 1][0], points[i + 1][coord_index]
        dt = max(t1 - t0, 1e-6)
        seg = (
            f"if(between(t,{_fmt(t0)},{_fmt(t1)}),"
            f"{_fmt(v0)}+({_fmt(v1)}-{_fmt(v0)})*(t-{_fmt(t0)})/{_fmt(dt)},"
        )
        expr += seg
    # Default value after last segment: hold last value
    v_last = points[-1][coord_index]
    expr += f"{_fmt(v_last)}" + (")" * (len(points) - 1))
    return expr


def _build_geq_mask_expr(points: List[Tuple[float, float, float]], erase_radius: int,
                         x_offset: float = 0.0, y_offset: float = 0.0) -> str:
    """
    Build a geq luminance expression that reveals in circular dabs along path points over time.
    For each point i: enable when t >= t_i, draw a filled circle of radius R centered at (x_i + x_offset, y_i + y_offset).
    We produce:
      max(255*gte(T,t0)*lte((X-x0)^2+(Y-y0)^2,R^2),
          255*gte(T,t1)*lte(...),
          ...)
    This allows accumulation (it never 'unreveal's).
    """
    terms = []
    r2 = erase_radius * erase_radius
    for t, x, y in points:
        xi = x + x_offset
        yi = y + y_offset
        term = (
            f"(255*gte(T,{_fmt(t)})*lte((X-{_fmt(int(round(xi)))})*(X-{_fmt(int(round(xi)))})"
            f"+(Y-{_fmt(int(round(yi)))})*(Y-{_fmt(int(round(yi)))})"
            f",{r2}))"
        )
        terms.append(term)
    # Nest max(...) to combine
    expr = terms[0]
    for term in terms[1:]:
        expr = f"max({expr},{term})"
    return expr


def _generate_path_points(
    pattern: str,
    width: int,
    height: int,
    wipe_start: float,
    wipe_duration: float,
    sample_points: int,
    center_x: float,
    center_y: float,
    radius_x: float,
    radius_y: float,
    amplitude: float,
) -> List[Tuple[float, float, float]]:
    """
    Generate (t, x, y) points for the given pattern.
    These match your original formulas (and are easy to extend).
    """
    pts: List[Tuple[float, float, float]] = []
    for i in range(sample_points):
        progress = i / (max(sample_points - 1, 1))
        t = wipe_start + progress * wipe_duration

        if pattern == "s_curve":
            # Vertical sweep top->bottom with horizontal oscillation
            y = height * 0.2 + (height * 0.6) * progress
            cycles = 2.5
            x = center_x + radius_x * 0.7 * math.sin(cycles * 2 * math.pi * progress)

        elif pattern == "ellipse":
            angle = 2 * math.pi * progress
            x = center_x + radius_x * math.cos(angle)
            y = center_y + amplitude * math.sin(angle)

        elif pattern == "vertical_sweep":
            # Top-to-bottom sweep with horizontal oscillation
            y = height * 0.15 + (height * 0.7) * progress
            x = center_x + radius_x * 0.5 * math.sin(4 * math.pi * progress)

        elif pattern == "figure8":
            # Figure-8 pattern for better vertical coverage
            angle = 2 * math.pi * progress
            x = center_x + radius_x * math.cos(angle)
            y = center_y + radius_y * math.sin(2 * angle)  # Double frequency for figure-8

        elif pattern == "triple_loop":
            # Three overlapping circular paths for head, torso, legs
            loop_idx = int(progress * 3)
            loop_progress = (progress * 3) % 1
            angle = 2 * math.pi * loop_progress
            
            if loop_idx == 0:  # Head area
                loop_center_y = height * 0.3
                loop_radius = radius_y * 0.6
            elif loop_idx == 1:  # Torso area
                loop_center_y = height * 0.5
                loop_radius = radius_y * 0.7
            else:  # Legs area
                loop_center_y = height * 0.7
                loop_radius = radius_y * 0.6
            
            x = center_x + radius_x * 0.8 * math.cos(angle)
            y = loop_center_y + loop_radius * math.sin(angle)

        else:
            # Fallback: simple center drift
            x = center_x
            y = center_y

        pts.append((t, x, y))
    return pts


def create_eraser_wipe(
    character_video: str,
    original_video: str,
    eraser_image: str,
    output_video: str,
    wipe_start: float = 0.0,
    wipe_duration: float = 0.6,
    mode: str = "true_erase",
    erase_radius: int = 120,
    sample_points: int = 25,
    path_pattern: str = "ellipse",
    # Graphic calibration: where is the eraser "tip" relative to the PNG?
    tip_x_ratio: float = 0.50,   # 0 = left edge, 1 = right edge
    tip_y_ratio: float = 0.12,   # 0 = top, 1 = bottom (your original 'tip_ratio')
    # Scale the eraser graphic
    scale_factor: float = 0.70,
    dry_run: bool = False
) -> bool:
    """
    Build and run an FFmpeg command that:
    1) Creates a time-accumulating circular-reveal mask along a chosen motion path,
    2) Mask-merges original video under the character video,
    3) Overlays the eraser PNG, whose tip follows THE SAME path (no hardcoded ellipse).
    
    Returns True on success, False on failure.
    """
    try:
        # Probe
        orig = _ffprobe_props(original_video)
        char = _ffprobe_props(character_video)
        # We will build mask at the output resolution of the original video
        width, height, fps = orig.width, orig.height, orig.fps

        # Motion params consistent with your code
        center_x = width / 2 - 20.0
        center_y = height / 2 + 10.0
        radius_x = 200.0
        radius_y = 150.0
        amplitude = radius_y * 0.8  # vertical amplitude used by some patterns

        # Scaled eraser size (the eraser can be taller than main; negative y is allowed)
        base_w, base_h = 768.0, 1344.0  # as in your notes
        scaled_w = int(round(base_w * scale_factor))
        scaled_h = int(round(base_h * scale_factor))

        # 1) Generate path points (t, x, y)
        points = _generate_path_points(
            pattern=path_pattern,
            width=width, height=height,
            wipe_start=wipe_start, wipe_duration=wipe_duration,
            sample_points=sample_points,
            center_x=center_x, center_y=center_y,
            radius_x=radius_x, radius_y=radius_y,
            amplitude=amplitude
        )
        assert len(points) >= 2, "path_points must contain at least 2 points"

        # 2) Build continuous overlay expressions (piecewise linear) from the SAME points
        x_tip_expr = _build_piecewise_linear_expr(points, coord_index=1)
        y_tip_expr = _build_piecewise_linear_expr(points, coord_index=2)

        # Align top-left of overlay so that eraser tip sits at (x(t), y(t))
        overlay_x_expr = f"({x_tip_expr}) - overlay_w*{_fmt(tip_x_ratio)}"
        overlay_y_expr = f"({y_tip_expr}) - overlay_h*{_fmt(tip_y_ratio)}"

        # 3) Build mask expression adding circular dabs over time at path points
        #    No broken Y "adjustment" — we use the actual (x, y) points.
        #    If your PNG tip is not exactly centered horizontally, you can offset here too (x_offset/y_offset).
        geq_luma_expr = _build_geq_mask_expr(points, erase_radius=erase_radius, x_offset=0.0, y_offset=0.0)

        # 4) Compose filter_complex
        #    Streams:
        #      [0:v] -> character
        #      [1:v] -> original
        #      [2:v] -> eraser PNG (looped), scaled to scaled_w x scaled_h
        #
        #    Steps:
        #      a) Base mask source: color black, gray format, fps=orig.fps, size=orig WxH
        #      b) geq -> build reveal mask in luminance plane
        #      c) maskedmerge: merge original UNDER character with mask (white => take from original)
        #      d) overlay eraser PNG following (overlay_x_expr, overlay_y_expr), only during wipe time
        #
        # Notes:
        #   - overlay: eval=frame is CRITICAL when expressions use 't'
        #   - enable=between(t, start, end) limits eraser visibility window
        #
        filter_parts = []

        # a) Loop & scale eraser PNG; ensure RGBA for overlay
        filter_parts.append(
            f"[2:v]loop=loop=999999:size=1:start=0,format=rgba,scale={scaled_w}:{scaled_h}[eraser]"
        )

        # b) Mask source + geq expression (quotes around lum_expr!)
        #    We rely on main fps to keep timing consistent with T in geq.
        lum_expr_quoted = geq_luma_expr.replace("'", r"\'")
        filter_parts.append(
            f"color=c=black@0.0:s={width}x{height}:r={_fmt(fps)}[m_src];"
            f"[m_src]format=gray,geq=lum='{lum_expr_quoted}'[mask]"
        )

        # c) maskedmerge: (first=input0 char), (second=input1 original), (third=mask)
        #    White mask => take from second => reveal original under character where erased
        filter_parts.append(
            f"[0:v]scale={width}:{height}:flags=bicubic,format=rgba[char];"
            f"[1:v]scale={width}:{height}:flags=bicubic,format=rgba[orig];"
            f"[char][orig][mask]maskedmerge[reveal]"
        )

        # d) Overlay eraser following the SAME path (no more hardcoded ellipse)
        #    Enable only during the wipe window (slightly extended by 0.02s if you like)
        enable_from = wipe_start
        enable_to = wipe_start + wipe_duration + 0.02
        ox = overlay_x_expr.replace("'", r"\'")
        oy = overlay_y_expr.replace("'", r"\'")
        filter_parts.append(
            f"[reveal][eraser]overlay="
            f"x='{ox}':y='{oy}':shortest=0:eof_action=pass:format=auto:eval=frame:"
            f"enable='between(t,{_fmt(enable_from)},{_fmt(enable_to)})'[outv]"
        )

        filter_complex = ";".join(filter_parts)

        # Build ffmpeg command
        # - Map video from [outv], audio from original if present
        cmd = [
            "ffmpeg", "-y", "-hide_banner",
            "-i", character_video,
            "-i", original_video,
            "-loop", "1", "-i", eraser_image,
            "-filter_complex", filter_complex,
            "-map", "[outv]",
            "-map", "1:a?",  # optional audio from original
            "-c:v", "libx264", "-preset", "medium", "-crf", "18",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            output_video
        ]

        # Debug output
        print(f"\n=== Fixed Eraser Wipe with {path_pattern} pattern ===")
        print(f"Resolution: {width}x{height} @ {fps:.3f} fps")
        print(f"Wipe window: [{wipe_start:.3f} .. {wipe_start + wipe_duration:.3f}] s")
        print(f"Scaled eraser: {scaled_w}x{scaled_h} (tip_x_ratio={tip_x_ratio}, tip_y_ratio={tip_y_ratio})")
        print("First 5 path points (t, x, y):")
        for i, (t, x, y) in enumerate(points[:5]):
            print(f"  {i:2d}: t={t:.3f}, x={x:.1f}, y={y:.1f}")
        print("Last 5 path points (t, x, y):")
        for i, (t, x, y) in enumerate(points[-5:], start=len(points)-5):
            print(f"  {i:2d}: t={t:.3f}, x={x:.1f}, y={y:.1f}")
        
        # Calculate Y coverage
        y_values = [y for _, _, y in points]
        y_min, y_max = min(y_values), max(y_values)
        print(f"\nY range covered: {y_min:.1f} to {y_max:.1f} (height: {y_max - y_min:.1f}px)")
        print(f"Coverage: {(y_max - y_min) / height * 100:.1f}% of video height")

        if dry_run:
            print("\nDry run only; not executing ffmpeg.")
            print("\nFFmpeg command:\n", " ".join(shlex.quote(c) for c in cmd))
            return True

        # Execute
        print("\nExecuting FFmpeg...")
        proc = _run(cmd)
        print(f"✓ Eraser wipe created successfully: {output_video}")
        return True
        
    except subprocess.CalledProcessError as e:
        print("✗ FFmpeg failed.")
        print(e.stderr.decode("utf-8", errors="ignore") if e.stderr else "No error details")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


if __name__ == "__main__":
    # Minimal CLI for ad-hoc testing
    import argparse
    p = argparse.ArgumentParser(description="Build an eraser-wipe effect with pattern-aware overlay motion.")
    p.add_argument("--character", required=True, help="Path to character video (foreground)")
    p.add_argument("--original", required=True, help="Path to original video (background)")
    p.add_argument("--eraser", required=True, help="Path to eraser PNG")
    p.add_argument("--out", required=True, help="Output video path")
    p.add_argument("--start", type=float, default=0.0, help="Wipe start time (seconds)")
    p.add_argument("--dur", type=float, default=0.6, help="Wipe duration (seconds)")
    p.add_argument("--pattern", default="s_curve", help="Path pattern (s_curve|ellipse|figure8|vertical_sweep|triple_loop)")
    p.add_argument("--radius", type=int, default=120, help="Erase radius (pixels)")
    p.add_argument("--samples", type=int, default=25, help="Sample points along the path")
    p.add_argument("--tipx", type=float, default=0.50, help="Tip x ratio in eraser image")
    p.add_argument("--tipy", type=float, default=0.12, help="Tip y ratio in eraser image")
    p.add_argument("--scale", type=float, default=0.70, help="Scale factor for eraser image")
    p.add_argument("--dry-run", action="store_true", help="Print command and exit")
    args = p.parse_args()

    create_eraser_wipe(
        character_video=args.character,
        original_video=args.original,
        eraser_image=args.eraser,
        output_video=args.out,
        wipe_start=args.start,
        wipe_duration=args.dur,
        path_pattern=args.pattern,
        erase_radius=args.radius,
        sample_points=args.samples,
        tip_x_ratio=args.tipx,
        tip_y_ratio=args.tipy,
        scale_factor=args.scale,
        dry_run=args.dry_run
    )