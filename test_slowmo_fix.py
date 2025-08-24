#!/usr/bin/env python3
"""Test the correct way to create slow-motion video."""

import subprocess
import tempfile
from pathlib import Path

video_path = "uploads/assets/videos/ai_math.mp4"
temp_dir = Path(tempfile.mkdtemp())
print(f"🧪 Testing slowmo in: {temp_dir}")

# Test different approaches to slow motion
gap_start = 4.02
gap_duration = 0.80
target_duration = 1.31  # We want to stretch 0.8s to 1.31s
speed_factor = gap_duration / target_duration  # 0.61

print(f"Goal: Take {gap_duration}s of video from {gap_start}s and stretch to {target_duration}s")
print(f"Speed factor: {speed_factor:.3f} ({speed_factor*100:.1f}%)")

# Method 1: Just setpts (WRONG - doesn't change duration)
seg1 = temp_dir / "seg_setpts_only.mp4"
pts_factor = 1.0 / speed_factor
cmd = [
    "ffmpeg", "-y", "-loglevel", "error",
    "-i", video_path,
    "-ss", str(gap_start), "-t", str(gap_duration),
    "-filter_complex", f"[0:v]setpts={pts_factor:.3f}*PTS[v]",
    "-map", "[v]", "-r", "30",
    "-c:v", "libx264", "-preset", "fast",
    "-pix_fmt", "yuv420p", "-an",
    str(seg1)
]
subprocess.run(cmd)

# Method 2: setpts with proper duration output
seg2 = temp_dir / "seg_setpts_fixed.mp4"
cmd = [
    "ffmpeg", "-y", "-loglevel", "error",
    "-i", video_path,
    "-ss", str(gap_start), "-to", str(gap_start + gap_duration),
    "-filter_complex", f"[0:v]setpts={pts_factor:.3f}*PTS[v]",
    "-map", "[v]",
    "-t", str(target_duration),  # Explicitly set output duration
    "-r", "30",
    "-c:v", "libx264", "-preset", "fast",
    "-pix_fmt", "yuv420p", "-an",
    str(seg2)
]
subprocess.run(cmd)

# Method 3: Use minterpolate for smooth slow motion
seg3 = temp_dir / "seg_minterpolate.mp4"
cmd = [
    "ffmpeg", "-y", "-loglevel", "error",
    "-i", video_path,
    "-ss", str(gap_start), "-t", str(gap_duration),
    "-filter_complex", f"[0:v]setpts={pts_factor:.3f}*PTS,minterpolate=fps=30[v]",
    "-map", "[v]",
    "-t", str(target_duration),
    "-c:v", "libx264", "-preset", "fast",
    "-pix_fmt", "yuv420p", "-an",
    str(seg3)
]
subprocess.run(cmd)

# Check durations
print("\n📊 Segment durations:")
for seg in [seg1, seg2, seg3]:
    if seg.exists():
        cmd = [
            "ffprobe", "-v", "error", "-show_entries",
            "format=duration", "-of", "default=noprint_wrappers=1:nokey=1",
            str(seg)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        try:
            duration = float(result.stdout.strip())
            status = "✅" if abs(duration - target_duration) < 0.1 else "❌"
            print(f"  {status} {seg.name}: {duration:.3f}s (target: {target_duration:.3f}s)")
        except:
            print(f"  ❌ {seg.name}: Failed to get duration")
    else:
        print(f"  ❌ {seg.name}: Not created")

print(f"\n📁 Results in: {temp_dir}")