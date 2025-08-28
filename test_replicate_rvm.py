#!/usr/bin/env python3
"""
Quick test of Robust Video Matting from Replicate
"""

import replicate
import requests
import subprocess
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

# Create 5-second test video
print("Creating 5-second test video...")
subprocess.run([
    "ffmpeg", "-y",
    "-i", "uploads/assets/videos/ai_math1.mp4",
    "-t", "5",
    "-c:v", "libx264", "-preset", "fast", "-crf", "18",
    "/tmp/ai_math1_5sec.mp4"
], check=True)

print("Running Robust Video Matting on Replicate...")
with open("/tmp/ai_math1_5sec.mp4", "rb") as f:
    output = replicate.run(
        "arielreplicate/robust_video_matting:73d2128a371922d5d1abf0712a1d974be0e4e2358cc1218e4e34714767232bac",
        input={"input_video": f}
    )

output_url = str(output)
print(f"Output URL: {output_url}")

# Download the result
print("Downloading matted video...")
response = requests.get(output_url, stream=True)
with open("/tmp/rvm_output.mp4", "wb") as f:
    for chunk in response.iter_content(chunk_size=8192):
        f.write(chunk)

# Composite with background
print("Compositing with background...")
subprocess.run([
    "ffmpeg", "-y",
    "-i", "uploads/assets/videos/ai_math1/ai_math1_background_0_0_5_0_STc2OalsLp.mp4",
    "-i", "/tmp/ai_math1_5sec.mp4",
    "-i", "/tmp/rvm_output.mp4",
    "-filter_complex",
    "[2:v]format=gray[mask];"
    "[0:v][1:v][mask]maskedmerge[out]",
    "-map", "[out]",
    "-c:v", "libx264", "-preset", "fast", "-crf", "23",
    "-pix_fmt", "yuv420p", "-movflags", "+faststart",
    "outputs/ai_math1_replicate_rvm.mp4"
], capture_output=True)

print("Done! Output: outputs/ai_math1_replicate_rvm.mp4")