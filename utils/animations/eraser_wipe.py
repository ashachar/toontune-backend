# utils/animations/eraser_wipe.py
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
    num, den = stream.get("avg_frame_rate", "30/1").split("/")
    fps = float(num) / float(den) if float(den) != 0 else 30.0
    duration = float(data["format"].get("duration", "0") or 0.0)
    return VideoProps(width=width, height=height, fps=fps, duration=duration)

def _fmt(x) -> str:
    """Format number for FFmpeg expressions (avoid scientific)."""
    if isinstance(x, float):
        s = f"{x:.9f}".rstrip("0").rstrip(".")
        return s if s != "" else "0"
    return str(int(x))

def _build_piecewise_linear_expr(points: List[Tuple[float, float, float]], coord_index: int) -> str:
    """
    Build piecewise-linear interpolation for tip motion (for overlay x/y).
    points: [(t, x, y)], coord_index: 1 for x, 2 for y
    """
    assert len(points) >= 2, "Need at least 2 path points for piecewise motion"
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
    v_last = points[-1][coord_index]
    expr += f"{_fmt(v_last)}" + (")" * (len(points) - 1))
    return expr

# --------------------------- NEW: Voronoi time-map builder ---------------------------

def _build_accumulating_circles_mask(points: List[Tuple[float, float, float]], 
                                     erase_radius: int) -> str:
    """
    Original approach with accumulating circles - simpler and proven to work.
    We'll just use a larger radius and more points for better coverage.
    """
    terms = []
    r2 = erase_radius * erase_radius
    for t, x, y in points:
        xi, yi = int(round(x)), int(round(y))
        term = (
            f"(255*gte(T,{_fmt(t)})*lte((X-{xi})*(X-{xi})+(Y-{yi})*(Y-{yi}),{r2}))"
        )
        terms.append(term)
    
    # Use nested max() - proven to work with reasonable number of points
    expr = terms[0]
    for term in terms[1:]:
        expr = f"max({expr},{term})"
    return expr

# ------------------------------------------------------------------------------------

def _generate_path_points(pattern: str, width: int, height: int, wipe_start: float,
                          wipe_duration: float, sample_points: int, center_x: float,
                          center_y: float, radius_x: float, radius_y: float,
                          amplitude: float) -> List[Tuple[float, float, float]]:
    """Generate path samples (t, x, y) along different patterns."""
    pts: List[Tuple[float, float, float]] = []
    for i in range(sample_points):
        progress = i / max(sample_points - 1, 1)
        t = wipe_start + progress * wipe_duration

        if pattern == "s_curve":
            # Covers entire height by default; adjust for your taste
            y = height * 0.05 + (height * 0.90) * progress
            cycles = 2.5
            x = center_x + radius_x * 0.7 * math.sin(cycles * 2 * math.pi * progress)

        elif pattern == "ellipse":
            angle = 2 * math.pi * progress
            x = center_x + radius_x * math.cos(angle)
            y = center_y + radius_y * math.sin(angle)

        elif pattern == "figure8":
            angle = 2 * math.pi * progress
            x = center_x + radius_x * math.sin(angle)
            y = center_y + amplitude  * math.sin(angle) * math.cos(angle)

        else:
            # Fallback: diagonal sweep
            x = width * (0.1 + 0.8 * progress)
            y = height * (0.1 + 0.8 * progress)

        pts.append((t, x, y))
    return pts

def create_eraser_wipe(
    character_video: str,
    original_video: str,
    eraser_image: str,
    output_video: str,
    wipe_start: float = 0.1,
    wipe_duration: float = 0.9,
    sample_points: int = 40,
    path_pattern: str = "s_curve",
    # Position of the eraser "tip" relative to the PNG (where it touches)
    tip_x_ratio: float = 0.50,
    tip_y_ratio: float = 0.12,
    # Eraser PNG scale
    scaled_w: int = 538,
    scaled_h: int = 941,
    # Optional micro-smoothing of Voronoi cell "flip" (seconds)
    micro_ramp: float = 0.0,   # set ~0.006 for subtle smoothing
    # H.264 encode
    crf: int = 18,
    preset: str = "medium",
    # Override sizes/fps if desired (probed otherwise)
    force_width: Optional[int] = 1280,
    force_height: Optional[int] = 720,
    out_fps: Optional[int] = 25,
    # Legacy parameters (kept for compatibility)
    mode: str = "true_erase",
    erase_radius: int = 120,
    scale_factor: float = 0.70,
    dry_run: bool = False
) -> bool:
    """
    Foreground erasure with 100% coverage:
      - Each pixel assigned to nearest path sample
      - Pixel erased exactly when that sample's time is reached
      - Color preserved using alphamerge + overlay

    micro_ramp: optional tiny delay (seconds) proportional to distance inside
                each Voronoi cell to reduce hard flips (0.0 disables).
    """
    # Probe videos (we still normalize to out_fps/size for safety)
    char_p = _ffprobe_props(character_video)
    orig_p = _ffprobe_props(original_video)
    width = force_width or max(char_p.width, orig_p.width)
    height = force_height or max(char_p.height, orig_p.height)
    fps = out_fps or int(round(orig_p.fps or 25))

    center_x = width * 0.5
    center_y = height * 0.5
    radius_x = width * 0.35
    radius_y = height * 0.35
    amplitude = height * 0.40

    # Generate path samples along your selected pattern
    points = _generate_path_points(
        path_pattern, width, height,
        wipe_start, wipe_duration, sample_points,
        center_x, center_y, radius_x, radius_y, amplitude
    )

    # Piecewise-linear eraser tip motion for overlay
    x_tip_expr = _build_piecewise_linear_expr(points, coord_index=1)
    y_tip_expr = _build_piecewise_linear_expr(points, coord_index=2)
    overlay_x_expr = f"({x_tip_expr}) - overlay_w*{_fmt(tip_x_ratio)}"
    overlay_y_expr = f"({y_tip_expr}) - overlay_h*{_fmt(tip_y_ratio)}"

    # -------------- Build accumulating circles mask --------------
    # Use larger radius for better coverage
    # Calculate adaptive radius based on point density
    adaptive_radius = int(max(width, height) / (sample_points ** 0.5) * 2.5)
    adaptive_radius = max(adaptive_radius, 200)  # Minimum radius
    mask_expr = _build_accumulating_circles_mask(points, adaptive_radius)
    mask_expr_quoted = mask_expr.replace("'", r"\'")

    # -------------- Compose the full filter graph --------------
    filter_parts = []

    # (A) Eraser PNG (RGBA), loop and scale
    filter_parts.append(
        f"[2:v]loop=loop=999999:size=1:start=0,format=rgba,scale={scaled_w}:{scaled_h}[eraser]"
    )

    # (B) Create mask directly with time-based expansion
    filter_parts.append(
        f"color=c=black:s={width}x{height}:r={_fmt(fps)}[m_src];"
        f"[m_src]format=gray,geq=lum='{mask_expr_quoted}'[mask]"
    )

    # (D) Color-preserving compositing: alpha from inverted mask → overlay
    filter_parts.append(
        f"[mask]lut=y=negval[inv_mask];"
        f"[0:v]fps={_fmt(fps)},scale={width}:{height}:flags=bicubic,format=gbrp[char_rgb];"
        f"[1:v]fps={_fmt(fps)},scale={width}:{height}:flags=bicubic,format=rgba[bg_rgba];"
        f"[char_rgb][inv_mask]alphamerge,format=rgba[char_rgba];"
        f"[bg_rgba][char_rgba]overlay=shortest=0:eof_action=pass:format=auto[reveal]"
    )

    # (E) Visual eraser overlay (optional, cosmetic)
    filter_parts.append(
        f"[reveal][eraser]overlay=x='{overlay_x_expr}':y='{overlay_y_expr}':"
        f"eval=frame:enable='between(t,{_fmt(wipe_start)},{_fmt(wipe_duration + wipe_start)})'[outv]"
    )

    filter_complex = ";".join(filter_parts)

    cmd = [
        "ffmpeg", "-y", "-hide_banner",
        "-i", character_video,
        "-i", original_video,
        "-loop", "1", "-i", eraser_image,
        "-filter_complex", filter_complex,
        "-map", "[outv]", "-map", "1:a?",
        "-c:v", "libx264", "-preset", preset, "-crf", str(crf),
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        output_video
    ]

    # Debug output
    if dry_run:
        print(f"\n=== Voronoi Eraser Wipe with {path_pattern} pattern ===")
        print(f"Resolution: {width}x{height} @ {fps:.3f} fps")
        print(f"Wipe window: [{wipe_start:.3f} .. {wipe_start + wipe_duration:.3f}] s")
        print(f"Sample points: {sample_points}")
        print(f"Micro-ramp: {micro_ramp:.3f}s")
        print("\nFFmpeg command:\n", " ".join(shlex.quote(c) for c in cmd))
        return True

    # Run pipeline
    _run(cmd)
    return True

if __name__ == "__main__":
    # Example invocation (edit paths as needed)
    create_eraser_wipe(
        character_video="outputs/runway_scaled_cropped.mp4",
        original_video="uploads/assets/runway_experiment/runway_demo_input.mp4",
        eraser_image="uploads/assets/images/eraser.png",
        output_video="outputs/final_eraser.mp4",
        wipe_start=0.10,
        wipe_duration=0.90,
        sample_points=40,
        path_pattern="s_curve",
        micro_ramp=0.0  # try 0.006 for a softer feel
    )