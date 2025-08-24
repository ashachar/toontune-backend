#!/usr/bin/env python3
"""Verify segments from a temp directory."""

import subprocess
import sys
from pathlib import Path

def get_video_duration(video_path):
    """Get duration of a video file."""
    cmd = [
        "ffprobe", "-v", "error", "-show_entries",
        "format=duration", "-of", "default=noprint_wrappers=1:nokey=1",
        str(video_path)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    try:
        return float(result.stdout.strip())
    except:
        return 0

if len(sys.argv) > 1:
    temp_dir = Path(sys.argv[1])
else:
    # Find most recent temp dir
    import tempfile
    import os
    temp_base = Path(tempfile.gettempdir())
    # List dirs starting with tmp
    temp_dirs = [d for d in temp_base.iterdir() if d.is_dir() and d.name.startswith('tmp')]
    if not temp_dirs:
        print("No temp directories found")
        sys.exit(1)
    # Get most recent
    temp_dir = max(temp_dirs, key=lambda d: d.stat().st_mtime)

print(f"📁 Checking segments in: {temp_dir}")
print("=" * 60)

# Find all segment files
segments = sorted(temp_dir.glob("seg_*.mp4"))

total_duration = 0
for seg in segments:
    duration = get_video_duration(seg)
    total_duration += duration
    print(f"{seg.name}: {duration:.3f}s")

print("-" * 60)
print(f"Total duration: {total_duration:.3f}s")

# Check concat file if it exists
concat_file = temp_dir / "concat.txt"
if concat_file.exists():
    print("\n📝 Concat file contents:")
    with open(concat_file) as f:
        print(f.read())

# Check final videos
for video in ["video_adjusted.mp4", "video_frozen.mp4"]:
    video_path = temp_dir / video
    if video_path.exists():
        duration = get_video_duration(video_path)
        print(f"\n🎬 {video}: {duration:.3f}s")