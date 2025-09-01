#!/usr/bin/env python3
"""
Regenerate mask for ai_math1b without compression.
"""

import sys
import os
import subprocess
import replicate
import requests
from pathlib import Path
from dotenv import load_dotenv

# Add parent directories to path
sys.path.append('../../')
sys.path.append('../../utils/video/background/')

load_dotenv()

# Define paths
input_video = "../../uploads/assets/videos/ai_math1b/raw_video.mp4"
output_dir = Path("../../uploads/assets/videos/ai_math1b")
output_dir.mkdir(exist_ok=True)

# First, extract 6 seconds from raw_video.mp4
temp_6sec = output_dir / "raw_video_6sec.mp4"
print(f"Extracting first 6 seconds from {input_video}...")
cmd = [
    "ffmpeg", "-y",
    "-i", input_video,
    "-t", "6",
    "-c:v", "copy",  # Copy without re-encoding
    "-c:a", "copy",
    str(temp_6sec)
]
subprocess.run(cmd, check=True)
print(f"Created: {temp_6sec}")

# Now call Replicate RVM to get green screen
print("\nCalling Replicate RVM for green screen mask...")
with open(temp_6sec, 'rb') as f:
    output = replicate.run(
        "arielreplicate/robust_video_matting:73d2128a371922d5d1abf0712a1d974be0e4e2358cc1218e4e34714767232bac",
        input={
            "input_video": f,
            "output_type": "green-screen"  # Get green screen output
        }
    )

output_url = str(output)
print(f"✓ Processing complete: {output_url}")

# Download the green screen result
print("Downloading green screen output...")
response = requests.get(output_url, stream=True)
response.raise_for_status()

# Save the raw green screen WITHOUT compression
temp_download = output_dir / "temp_green_screen.mp4"
with open(temp_download, 'wb') as f:
    for chunk in response.iter_content(chunk_size=8192):
        f.write(chunk)

print(f"Downloaded to: {temp_download}")

# Now save with LOSSLESS encoding to preserve exact colors
output_mask = output_dir / "raw_video_6sec_mask_lossless.mp4"
print("\nSaving with lossless encoding to preserve exact green colors...")
cmd = [
    "ffmpeg", "-y",
    "-i", str(temp_download),
    "-c:v", "libx264",
    "-crf", "0",  # Lossless!
    "-preset", "veryslow",  # Best compression for lossless
    "-pix_fmt", "yuv444p",  # Full color resolution (no chroma subsampling)
    str(output_mask)
]
subprocess.run(cmd, check=True)

# Also save as raw frames (truly uncompressed)
frames_dir = output_dir / "mask_frames"
frames_dir.mkdir(exist_ok=True)
print(f"\nExtracting raw frames to {frames_dir}...")
cmd = [
    "ffmpeg", "-y",
    "-i", str(temp_download),
    str(frames_dir / "frame_%04d.png")  # PNG is lossless
]
subprocess.run(cmd, check=True)

# Clean up temp download
temp_download.unlink()

print(f"\n✅ Done!")
print(f"Lossless video: {output_mask}")
print(f"Raw frames: {frames_dir}")
print("\nNow let's check for green variation...")