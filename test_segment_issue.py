#!/usr/bin/env python3
"""Test to reproduce the segment issue."""

import subprocess
import tempfile
from pathlib import Path

# Use the ai_math video
video_path = "uploads/assets/videos/ai_math.mp4"

# Create temp dir
temp_dir = Path(tempfile.mkdtemp())
print(f"🧪 Testing in: {temp_dir}")

# Create segments exactly as the pipeline would
segments = []

# Segment 0: Normal [0.00s → 4.02s]
seg0 = temp_dir / "seg_000_normal.mp4"
cmd = [
    "ffmpeg", "-y", "-loglevel", "warning",
    "-i", video_path,
    "-ss", "0", "-t", "4.02",
    "-c:v", "libx264", "-preset", "fast", "-crf", "23",
    "-pix_fmt", "yuv420p", "-an",
    str(seg0)
]
subprocess.run(cmd)
segments.append(seg0)
print(f"✅ Created segment 0: 0.00s → 4.02s")

# Segment 1: Slowed [4.02s for 0.80s] → 1.31s @ 61.3%
seg1 = temp_dir / "seg_001_slow.mp4"
pts_factor = 1.0 / 0.613
cmd = [
    "ffmpeg", "-y", "-loglevel", "warning",
    "-i", video_path,
    "-ss", "4.02", "-t", "0.80",
    "-filter_complex", f"[0:v]setpts={pts_factor:.3f}*PTS[v]",
    "-map", "[v]", "-r", "30",
    "-c:v", "libx264", "-preset", "fast", "-crf", "23",
    "-pix_fmt", "yuv420p", "-an",
    str(seg1)
]
subprocess.run(cmd)
segments.append(seg1)
print(f"✅ Created segment 1: 4.02s for 0.80s slowed to ~1.31s")

# Segment 2: Normal [4.82s → 31.54s]
seg2 = temp_dir / "seg_002_normal.mp4"
cmd = [
    "ffmpeg", "-y", "-loglevel", "warning",
    "-i", video_path,
    "-ss", "4.82", "-t", "26.72",
    "-c:v", "libx264", "-preset", "fast", "-crf", "23",
    "-pix_fmt", "yuv420p", "-an",
    str(seg2)
]
subprocess.run(cmd)
segments.append(seg2)
print(f"✅ Created segment 2: 4.82s → 31.54s")

# Check segment durations
print("\n📊 Segment durations:")
for seg in segments:
    cmd = [
        "ffprobe", "-v", "error", "-show_entries",
        "format=duration", "-of", "default=noprint_wrappers=1:nokey=1",
        str(seg)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    duration = float(result.stdout.strip())
    print(f"  {seg.name}: {duration:.3f}s")

# Concatenate segments
concat_file = temp_dir / "concat.txt"
with open(concat_file, "w") as f:
    for seg in segments:
        f.write(f"file '{seg.absolute()}'\n")

output = temp_dir / "test_output.mp4"
cmd = [
    "ffmpeg", "-y", "-loglevel", "warning",
    "-f", "concat", "-safe", "0",
    "-i", str(concat_file),
    "-c:v", "libx264", "-preset", "fast", "-crf", "23",
    "-pix_fmt", "yuv420p",
    str(output)
]
subprocess.run(cmd)
print(f"\n✅ Created concatenated video: {output}")

# Extract frames at key points
print("\n🖼️ Extracting frames for comparison:")
for t in [25.0, 30.0, 31.0, 32.0]:
    # From concatenated video
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(output),
        "-ss", str(t), "-vframes", "1",
        str(temp_dir / f"concat_{t}.png")
    ]
    subprocess.run(cmd)
    
    # From original video
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", video_path,
        "-ss", str(t), "-vframes", "1",
        str(temp_dir / f"orig_{t}.png")
    ]
    subprocess.run(cmd)
    print(f"  Extracted frames at {t}s")

print(f"\n📁 Results in: {temp_dir}")
print("Compare concat_*.png with orig_*.png to see the issue")