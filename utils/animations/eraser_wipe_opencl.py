# utils/animations/eraser_wipe_opencl.py
import math
import os
import json
import shlex
import subprocess
import tempfile
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
                          sample_points: int,
                          center_x: float, center_y: float,
                          radius_x: float, radius_y: float,
                          amplitude: float = 0.0) -> List[Tuple[float, float, float]]:
    """Generate (t_norm, x, y) samples along a path."""
    pts: List[Tuple[float, float, float]] = []
    for i in range(sample_points):
        prog = i / max(sample_points - 1, 1)
        t_norm = prog  # normalized 0..1 along wipe duration

        if pattern == "s_curve":
            # S-curve covering ~90% of frame height with horizontal oscillation
            y = height * 0.05 + (height * 0.90) * prog
            cycles = 2.5
            x = center_x + radius_x * 0.7 * math.sin(cycles * 2 * math.pi * prog)
        elif pattern == "ellipse":
            ang = 2 * math.pi * prog
            x = center_x + radius_x * math.cos(ang)
            y = center_y + radius_y * math.sin(ang)
        else:
            # default: straight vertical sweep center
            x = center_x
            y = height * 0.05 + (height * 0.90) * prog

        pts.append((t_norm, x, y))
    return pts

def _expr_from_lookup(values_0to255: List[int]) -> str:
    """
    Build a tiny geq expression f(X) that returns values_0to255[X] for X in [0..K-1].
    This runs over a 1xK image only, so nested if is fine (K <= ~128 is trivial).
    """
    expr = "0"
    for i in reversed(range(len(values_0to255))):
        expr = f"if(eq(X,{i}),{values_0to255[i]},{expr})"
    return expr

def _overlay_xy_expr(pattern: str, wipe_start: float, wipe_duration: float,
                     width: int, height: int, eraser_w: int, eraser_h: int,
                     center_x: float, center_y: float, radius_x: float, radius_y: float) -> Tuple[str, str]:
    # progress: clip((t - wipe_start)/wipe_duration, 0, 1)
    prog = f"clip((t-{wipe_start})/{wipe_duration},0,1)"
    if pattern == "s_curve":
        x = f"{center_x} + {radius_x*0.7}*sin(2*PI*2.5*{prog}) - {eraser_w}/2"
        y = f"{height*0.05} + {height*0.90}*{prog} - {eraser_h}/2"
    elif pattern == "ellipse":
        x = f"{center_x} + {radius_x}*cos(2*PI*{prog}) - {eraser_w}/2"
        y = f"{center_y} + {radius_y}*sin(2*PI*{prog}) - {eraser_h}/2"
    else:
        x = f"{center_x} - {eraser_w}/2"
        y = f"{height*0.05} + {height*0.90}*{prog} - {eraser_h}/2"
    return x, y

def create_eraser_wipe_opencl(
    character_video: str,
    original_video: str,
    eraser_image: str,
    output_video: str,
    wipe_start: float = 0.10,
    wipe_duration: float = 0.90,
    sample_points: int = 40,
    path_pattern: str = "s_curve",
    # sizes/defaults will be probed from videos
    eraser_scale: float = 0.12,  # relative to min(width,height)
    quality_crf: int = 18,
    x264_preset: str = "medium",
) -> None:
    """
    Single-FFmpeg-command pipeline that:
      * Builds a 1xK points texture (R=x_norm, G=y_norm, B=t_norm),
      * Computes a per-pixel nearest-neighbor time map in one OpenCL pass,
      * Loops that single tmap frame,
      * Erases foreground pixels when T >= tmap(x,y),
      * Preserves full color via alphamerge,
      * Overlays the eraser PNG following the exact same path.
    """

    # 1) Probe sizes and fps
    w0, h0, fps0 = _probe_video_dims(character_video)
    w1, h1, fps1 = _probe_video_dims(original_video)
    width  = max(w0, w1)
    height = max(h0, h1)
    fps    = round(max(fps0, fps1) or 25.0)

    # 2) Choose eraser visual size
    base = min(width, height)
    eraser_w = int(round(base * eraser_scale))
    eraser_h = eraser_w  # square eraser

    # 3) Generate path samples (normalized time, absolute x,y)
    cx, cy = width/2.0, height/2.0
    rx, ry = width*0.35, height*0.35
    pts = _generate_path_points(path_pattern, width, height,
                                wipe_start, wipe_duration, sample_points,
                                cx, cy, rx, ry)

    # 4) Build the 1xK RGBA row (0..255 per channel)
    #    R = x_norm, G = y_norm, B = t_norm, A=255
    xs = [max(0, min(255, int(round(255.0 * (x / max(1, width)) ))))  for (_, x, _) in pts]
    ys = [max(0, min(255, int(round(255.0 * (y / max(1, height)))))) for (_, _, y) in pts]
    ts = [max(0, min(255, int(round(255.0 * t))))                     for (t, _, _) in pts]
    rs, gs, bs = _expr_from_lookup(xs), _expr_from_lookup(ys), _expr_from_lookup(ts)
    as_ = "255"

    # 5) Overlay motion expressions synchronized with the exact same path param
    ox_expr, oy_expr = _overlay_xy_expr(path_pattern, wipe_start, wipe_duration,
                                        width, height, eraser_w, eraser_h,
                                        cx, cy, rx, ry)

    # 6) Create the OpenCL kernel file (once)
    kernel_code = r"""
__kernel void tmap(__write_only image2d_t dst,
                   unsigned int index,
                   __read_only  image2d_t pts)
{
    const sampler_t s = (CLK_NORMALIZED_COORDS_FALSE |
                         CLK_ADDRESS_CLAMP |
                         CLK_FILTER_NEAREST);
    int2 loc = (int2)(get_global_id(0), get_global_id(1));
    int2 dsz = get_image_dim(dst);
    int  W   = dsz.x, H = dsz.y;
    int  K   = get_image_dim(pts).x;

    float px = (float)loc.x + 0.5f;
    float py = (float)loc.y + 0.5f;

    float best_d2 = FLT_MAX;
    float best_t  = 0.0f;

    for (int i = 0; i < K; ++i) {
        float4 p = read_imagef(pts, s, (int2)(i,0));
        float cx = p.x * (float)W;
        float cy = p.y * (float)H;
        float tn = p.z;
        float dx = px - cx;
        float dy = py - cy;
        float d2 = dx*dx + dy*dy;
        if (d2 < best_d2) { best_d2 = d2; best_t = tn; }
    }
    float4 outv = (float4)(best_t, best_t, best_t, 1.0f);
    write_imagef(dst, loc, outv);
}
""".strip("\n")

    with tempfile.TemporaryDirectory() as tdir:
        kernel_path = os.path.join(tdir, "eraser_tmap.cl")
        with open(kernel_path, "w", encoding="utf-8") as f:
            f.write(kernel_code)

        # 7) Compose the filter_complex
        #    - Build points texture [pts1] as a single frame, then loop it.
        #    - Run program_opencl once to build [tmap1] single frame, then loop it.
        #    - Make mask at each time T: pixels with tmap <= progress(T) become transparent.
        #      Use a small softness (ease) of ~1 frame for a smooth, professional look.
        ease = 1.2 / fps
        progress = f"clip((T-{wipe_start})/{wipe_duration},0,1)"
        # Use 8-bit tmap so p(X,Y)/255 gives t_norm
        mask_expr = f"255*clip(({progress} - p(X,Y)/255)/{ease} + 0.5, 0, 1)"

        filter_parts = []

        # Inputs: [0:char], [1:bg], [2:eraser]
        # Points row (1xK), single frame -> loop
        filter_parts += [
            f"color=c=black:s={sample_points}x1:r=1[pts_src]",
            f"[pts_src]format=rgba,geq=r='{rs}':g='{gs}':b='{bs}':a='{as_}',trim=end_frame=1[pts1]",
            f"[pts1]loop=loop=1000000:size=1:start=0[pts]"
        ]

        # Generate time map once via OpenCL (size WxH), then loop it
        filter_parts += [
            f"[pts]format=rgba,hwupload[pts_cl]",
            f"[pts_cl]program_opencl=source='{kernel_path}':kernel='tmap':size={width}x{height}[tmap_cl]",
            f"[tmap_cl]hwdownload,format=gray,trim=end_frame=1[tmap1]",
            f"[tmap1]loop=loop=1000000:size=1:start=0[tmap]"
        ]

        # Foreground and background preprocessing; color-preserving composite
        filter_parts += [
            f"[0:v]fps={fps},scale={width}:{height}:flags=bicubic,format=gbrp[char_rgb]",
            f"[1:v]fps={fps},scale={width}:{height}:flags=bicubic,format=rgba[bg_rgba]",
            f"[tmap]geq=lum='{mask_expr}',format=gray[mask]",  # alpha = inverse mask
            f"[mask]lut=y=negval[inv_mask]",
            f"[char_rgb][inv_mask]alphamerge,format=rgba[char_rgba]",
            f"[bg_rgba][char_rgba]overlay=shortest=0:eof_action=pass:format=auto[reveal]"
        ]

        # Eraser overlay follows the same path
        filter_parts += [
            f"[2:v]loop=loop=999999:size=1:start=0,format=rgba,scale={eraser_w}:{eraser_h}[eraser]",
            (
                f"[reveal][eraser]overlay="
                f"x='{ox_expr}':y='{oy_expr}':eval=frame:"
                f"enable='between(t,{wipe_start},{wipe_start + wipe_duration})'[outv]"
            )
        ]

        filter_complex = ";".join(filter_parts)

        # 8) Build and run the ffmpeg command
        cmd = [
            "ffmpeg", "-y", "-hide_banner",
            # Initialize OpenCL device for program_opencl
            "-init_hw_device", "opencl=gpu:0",
            "-filter_hw_device", "gpu",

            "-i", character_video,
            "-i", original_video,
            "-loop", "1", "-i", eraser_image,

            "-filter_complex", filter_complex,
            "-map", "[outv]", "-map", "1:a?",
            "-c:v", "libx264", "-preset", x264_preset, "-crf", str(quality_crf),
            "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            output_video,
        ]

        print("Running:\n", " ".join(shlex.quote(x) for x in cmd))
        subprocess.run(cmd, check=True)

if __name__ == "__main__":
    # Example run (adjust paths as needed):
    create_eraser_wipe_opencl(
        character_video="outputs/runway_scaled_cropped.mp4",
        original_video="uploads/assets/runway_experiment/runway_demo_input.mp4",
        eraser_image="uploads/assets/images/eraser.png",
        output_video="outputs/test_opencl.mp4",
        wipe_start=0.10,
        wipe_duration=0.90,
        sample_points=40,
        path_pattern="s_curve",
        eraser_scale=0.12
    )