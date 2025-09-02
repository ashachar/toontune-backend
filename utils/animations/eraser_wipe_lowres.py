# utils/animations/eraser_wipe_lowres.py
"""
CPU-only fallback: Compute Voronoi time map at low resolution with geq,
then upscale to full resolution. Still single FFmpeg command.
"""

import math
import json
import shlex
import subprocess
from typing import List, Tuple

def _probe_video_dims(path: str) -> Tuple[int, int, float]:
    """Return (width, height, fps) from ffprobe."""
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,avg_frame_rate",
        "-of", "json", path
    ]
    out = subprocess.check_output(cmd).decode("utf-8")
    info = json.loads(out)["streams"][0]
    w, h = info["width"], info["height"]
    num, den = map(int, info["avg_frame_rate"].split("/"))
    fps = num / den if den else 25.0
    return w, h, fps

def _generate_path_points(pattern: str,
                          width: int, height: int,
                          wipe_start: float, wipe_duration: float,
                          sample_points: int) -> List[Tuple[float, float, float]]:
    """Generate (t, x, y) samples along a path."""
    pts: List[Tuple[float, float, float]] = []
    center_x = width * 0.5
    center_y = height * 0.5
    radius_x = width * 0.35
    radius_y = height * 0.35
    
    for i in range(sample_points):
        prog = i / max(sample_points - 1, 1)
        t = wipe_start + prog * wipe_duration

        if pattern == "s_curve":
            # S-curve covering ~90% of frame height
            y = height * 0.05 + (height * 0.90) * prog
            cycles = 2.5
            x = center_x + radius_x * 0.7 * math.sin(cycles * 2 * math.pi * prog)
        elif pattern == "ellipse":
            ang = 2 * math.pi * prog
            x = center_x + radius_x * math.cos(ang)
            y = center_y + radius_y * math.sin(ang)
        else:
            # vertical sweep
            x = center_x
            y = height * 0.05 + (height * 0.90) * prog

        pts.append((t, x, y))
    return pts

def _build_lowres_voronoi_expr(points: List[Tuple[float, float, float]], 
                               width: int, height: int,
                               lowres_w: int, lowres_h: int,
                               wipe_start: float, wipe_duration: float) -> str:
    """
    Build a Voronoi expression for low-res computation using st()/ld() registers.
    Returns normalized time (0-255) for nearest point.
    """
    if not points:
        return "0"
    
    # Scale points to lowres coordinates
    scale_x = lowres_w / width
    scale_y = lowres_h / height
    
    t0, x0, y0 = points[0]
    x0_lr = int(round(x0 * scale_x))
    y0_lr = int(round(y0 * scale_y))
    
    # Initialize with first point
    d0 = f"((X-{x0_lr})*(X-{x0_lr})+(Y-{y0_lr})*(Y-{y0_lr}))"
    t0_norm = (t0 - wipe_start) / wipe_duration
    init = f"st(0,{d0})*0+st(1,{t0_norm})*0"
    
    # Update for each subsequent point
    updates = []
    for t, x, y in points[1:]:
        x_lr = int(round(x * scale_x))
        y_lr = int(round(y * scale_y))
        t_norm = (t - wipe_start) / wipe_duration
        
        di = f"((X-{x_lr})*(X-{x_lr})+(Y-{y_lr})*(Y-{y_lr}))"
        # If this point is closer, update both distance and time
        updates.append(
            f"st(1,if(lt({di},ld(0)),{t_norm},ld(1)))*0+"
            f"st(0,if(lt({di},ld(0)),{di},ld(0)))*0"
        )
    
    # Build expression: init + updates + final value
    expr = init + "+" + "+".join(updates) if updates else init
    # Return normalized time as 0-255
    expr = f"({expr})+255*clip(ld(1),0,1)"
    
    return expr

def _build_piecewise_motion(points: List[Tuple[float, float, float]], 
                           coord_idx: int) -> str:
    """Build piecewise linear interpolation for eraser motion."""
    if len(points) < 2:
        return str(points[0][coord_idx] if points else 0)
    
    expr = ""
    for i in range(len(points) - 1):
        t0, v0 = points[i][0], points[i][coord_idx]
        t1, v1 = points[i+1][0], points[i+1][coord_idx]
        dt = max(t1 - t0, 0.001)
        
        expr += (
            f"if(between(t,{t0},{t1}),"
            f"{v0}+({v1}-{v0})*(t-{t0})/{dt},"
        )
    
    # Last value
    expr += str(points[-1][coord_idx])
    expr += ")" * (len(points) - 1)
    
    return expr

def create_eraser_wipe_lowres(
    character_video: str,
    original_video: str,
    eraser_image: str,
    output_video: str,
    wipe_start: float = 0.10,
    wipe_duration: float = 0.90,
    sample_points: int = 30,
    path_pattern: str = "s_curve",
    lowres_scale: float = 0.125,  # 1/8 resolution for time map
    eraser_scale_w: int = 538,
    eraser_scale_h: int = 941,
    tip_x_ratio: float = 0.5,
    tip_y_ratio: float = 0.12,
    crf: int = 18,
    preset: str = "medium"
) -> None:
    """
    CPU-only version: Compute Voronoi at low resolution, upscale, and apply.
    Single FFmpeg command, no external dependencies.
    """
    
    # Probe video dimensions
    w0, h0, fps0 = _probe_video_dims(character_video)
    w1, h1, fps1 = _probe_video_dims(original_video)
    width = max(w0, w1)
    height = max(h0, h1)
    fps = round(max(fps0, fps1) or 25.0)
    
    # Low-res dimensions for time map computation
    lowres_w = max(32, int(width * lowres_scale))
    lowres_h = max(32, int(height * lowres_scale))
    
    print(f"Computing time map at {lowres_w}x{lowres_h}, upscaling to {width}x{height}")
    
    # Generate path points
    points = _generate_path_points(path_pattern, width, height, 
                                  wipe_start, wipe_duration, sample_points)
    
    # Build Voronoi expression for low-res
    voronoi_expr = _build_lowres_voronoi_expr(
        points, width, height, lowres_w, lowres_h, wipe_start, wipe_duration
    )
    voronoi_expr_quoted = voronoi_expr.replace("'", r"\'")
    
    # Build eraser motion expressions
    x_expr = _build_piecewise_motion(points, 1)
    y_expr = _build_piecewise_motion(points, 2)
    overlay_x = f"({x_expr})-overlay_w*{tip_x_ratio}"
    overlay_y = f"({y_expr})-overlay_h*{tip_y_ratio}"
    
    # Build filter_complex
    filter_parts = []
    
    # 1. Loop eraser PNG
    filter_parts.append(
        f"[2:v]loop=loop=999999:size=1:start=0,format=rgba,"
        f"scale={eraser_scale_w}:{eraser_scale_h}[eraser]"
    )
    
    # 2. Compute low-res time map ONCE, upscale, and loop
    filter_parts.append(
        f"color=c=black:s={lowres_w}x{lowres_h}:r=1[lr_src];"
        f"[lr_src]format=gray,geq=lum='{voronoi_expr_quoted}',trim=end_frame=1[lr_tmap];"
        f"[lr_tmap]scale={width}:{height}:flags=bicubic[hr_tmap];"
        f"[hr_tmap]loop=loop=1000000:size=1:start=0,setpts=N/{fps}/TB[tmap]"
    )
    
    # 3. Create time-based mask
    progress = f"clip((T-{wipe_start})/{wipe_duration},0,1)"
    ease = 1.0 / fps  # 1 frame smoothing
    mask_expr = f"255*clip(({progress}-p(X,Y)/255)/{ease}+0.5,0,1)"
    
    filter_parts.append(
        f"[tmap]geq=lum='{mask_expr}'[mask]"
    )
    
    # 4. Color-preserving composite
    filter_parts.append(
        f"[mask]lut=y=negval[inv_mask];"
        f"[0:v]fps={fps},scale={width}:{height}:flags=bicubic,format=gbrp[char_rgb];"
        f"[1:v]fps={fps},scale={width}:{height}:flags=bicubic,format=rgba[bg_rgba];"
        f"[char_rgb][inv_mask]alphamerge,format=rgba[char_rgba];"
        f"[bg_rgba][char_rgba]overlay=shortest=0:eof_action=pass:format=auto[reveal]"
    )
    
    # 5. Overlay eraser
    filter_parts.append(
        f"[reveal][eraser]overlay="
        f"x='{overlay_x}':y='{overlay_y}':eval=frame:"
        f"enable='between(t,{wipe_start},{wipe_start + wipe_duration})'[outv]"
    )
    
    filter_complex = ";".join(filter_parts)
    
    # Run FFmpeg
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
    
    print("Running command...")
    print(" ".join(shlex.quote(x) for x in cmd[:5]) + " ...")
    subprocess.run(cmd, check=True)
    print(f"✓ Created: {output_video}")

if __name__ == "__main__":
    create_eraser_wipe_lowres(
        character_video="outputs/runway_scaled_cropped.mp4",
        original_video="uploads/assets/runway_experiment/runway_demo_input.mp4",
        eraser_image="uploads/assets/images/eraser.png",
        output_video="outputs/final_eraser_lowres.mp4",
        wipe_start=0.10,
        wipe_duration=0.90,
        sample_points=30,
        path_pattern="s_curve",
        lowres_scale=0.125  # 160x90 for 1280x720
    )