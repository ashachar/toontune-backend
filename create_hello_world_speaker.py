#!/usr/bin/env python3
"""Create Hello World animation on the actual AI speaker video using refactored module."""

import subprocess
import sys
import os

print("="*70)
print("Creating 'Hello World' animation on AI speaker video")
print("Using the refactored letter_3d_dissolve module")
print("="*70)

input_video = "uploads/assets/videos/ai_math1_4sec.mp4"
output_video = "outputs/hello_world_speaker_refactored.mp4"
output_h264 = "outputs/hello_world_speaker_refactored_h264.mp4"

print(f"\n📹 Input: {input_video}")
print(f"📝 Text: 'Hello World'")
print(f"📁 Output: {output_h264}")

# Run the animation with the refactored module
cmd = [
    "python", "-m", "utils.animations.apply_3d_text_animation",
    input_video,
    "--text", "Hello World",
    "--position", "640,400",  # Center position for 1280x720
    "--motion-duration", "0.8",
    "--dissolve-duration", "2.5", 
    "--output", output_video,
    "--supersample", "8",  # High quality
]

print("\nRunning animation pipeline with refactored module...")
print("-"*50)

result = subprocess.run(cmd, capture_output=True, text=True)

if result.returncode == 0:
    print("✅ Animation created successfully!")
    
    # Convert to H.264 for compatibility
    print("\nConverting to H.264 format...")
    convert_cmd = [
        "ffmpeg", "-i", output_video,
        "-c:v", "libx264", "-preset", "slow", "-crf", "18",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        "-y", output_h264
    ]
    
    convert_result = subprocess.run(convert_cmd, capture_output=True, text=True)
    
    if convert_result.returncode == 0:
        print(f"✅ H.264 version created: {output_h264}")
        
        # Get file size
        size_mb = os.path.getsize(output_h264) / (1024 * 1024)
        print(f"📊 File size: {size_mb:.2f} MB")
        
        print("\n" + "="*70)
        print("✅ SUCCESS!")
        print("="*70)
        print(f"\n🎬 Video created: {output_h264}")
        print("\nFeatures demonstrated:")
        print("  • Real AI speaker from ai_math1.mp4")
        print("  • 'Hello World' text with 3D effect")
        print("  • Motion phase (0.8s) - text emerges with depth")
        print("  • Dissolve phase (2.5s) - letter-by-letter fade")
        print("  • Dynamic occlusion - text behind speaker")
        print("\nUsing refactored module structure:")
        print("  • letter_3d_dissolve/dissolve.py")
        print("  • letter_3d_dissolve/timing.py") 
        print("  • letter_3d_dissolve/renderer.py")
        print("  • letter_3d_dissolve/occlusion.py")
        print("  • ...and 4 more modules")
        print("\n👀 Open the video to see the result!")
        
    else:
        print(f"❌ H.264 conversion failed: {convert_result.stderr}")
        
else:
    print(f"❌ Animation creation failed")
    if result.stderr:
        print(f"Error: {result.stderr}")
    sys.exit(1)