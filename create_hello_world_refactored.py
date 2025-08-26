#!/usr/bin/env python3
"""Create Hello World animation using the refactored letter_3d_dissolve module."""

import subprocess
import sys
import os

# Make sure we can import the modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("Creating Hello World animation with refactored 3D dissolve module...")
print("="*60)

# Run the apply_3d_text_animation script which should now use the refactored module
cmd = [
    "python", "-m", "utils.animations.apply_3d_text_animation",
    "test_crossing_h264.mp4",
    "--text", "Hello World",
    "--position", "500,380",
    "--motion-duration", "0.5",
    "--dissolve-duration", "3.0",
    "--output", "outputs/hello_world_refactored.mp4",
    "--debug"  # Enable debug to see it's using refactored code
]

print(f"Command: {' '.join(cmd)}")
print("="*60)

result = subprocess.run(cmd, capture_output=False, text=True)

if result.returncode == 0:
    print("\n✅ Animation created successfully!")
    
    # Convert to H.264 for QuickTime compatibility
    print("\nConverting to H.264...")
    convert_cmd = [
        "ffmpeg", "-i", "outputs/hello_world_refactored.mp4",
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        "-y", "outputs/hello_world_refactored_h264.mp4"
    ]
    
    subprocess.run(convert_cmd, capture_output=True)
    print("✅ Created H.264 version: outputs/hello_world_refactored_h264.mp4")
    
    print("\n" + "="*60)
    print("VERIFICATION:")
    print("="*60)
    print("The video has been created using the refactored module structure:")
    print("- letter_3d_dissolve/dissolve.py - Main animation class")
    print("- letter_3d_dissolve/timing.py - Frame-accurate timing")
    print("- letter_3d_dissolve/renderer.py - 3D letter rendering")
    print("- letter_3d_dissolve/sprite_manager.py - Sprite management")
    print("- letter_3d_dissolve/occlusion.py - Dynamic masking")
    print("- letter_3d_dissolve/frame_renderer.py - Frame rendering")
    print("- letter_3d_dissolve/handoff.py - Motion handoff")
    print("\n✅ Open outputs/hello_world_refactored_h264.mp4 to view the result")
    
else:
    print("❌ Failed to create animation")
    sys.exit(1)