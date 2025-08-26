#!/usr/bin/env python3
"""Create Hello World demo with the original walking person video."""

import subprocess
import os

# Use the test crossing video that has the walking person
input_video = "test_crossing_h264.mp4"

# Verify it exists
if not os.path.exists(input_video):
    print(f"❌ Video not found: {input_video}")
    exit(1)

output_video = "outputs/hello_world_with_occlusion.mp4"

# Create the animation with Hello World text
cmd = [
    "python", "utils/animations/apply_3d_text_animation.py",
    input_video,
    "Hello World",
    "--position", "500,380",
    "--motion-duration", "0.5",
    "--dissolve-duration", "3.0",
    "--output", output_video
]

print(f"Creating Hello World animation with occlusion...")
print(f"Input: {input_video}")
print(f"Text: 'Hello World'")
print(f"Position: (500, 380)")
print(f"Motion: 0.5s, Dissolve: 3.0s")
print()

result = subprocess.run(cmd, capture_output=False, text=True)

if result.returncode == 0:
    print(f"\n✅ Created: {output_video}")
    
    # Convert to H.264 for compatibility
    h264_output = output_video.replace(".mp4", "_h264.mp4")
    convert_cmd = [
        "ffmpeg", "-i", output_video,
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        "-y", h264_output
    ]
    
    print(f"\nConverting to H.264 for QuickTime compatibility...")
    subprocess.run(convert_cmd, capture_output=True)
    print(f"✅ Created H.264 version: {h264_output}")
    
else:
    print(f"❌ Failed to create animation")
    exit(1)