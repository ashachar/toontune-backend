#!/usr/bin/env python3
"""Extract segment from 2:20-2:24 and apply Hello World animation with refactored module."""

import subprocess
import os
import sys

print("="*80)
print("🎬 CREATING HELLO WORLD ANIMATION ON VIDEO SEGMENT (2:20-2:24)")
print("="*80)

# Step 1: Extract the 4-second segment from the original video
input_video = "uploads/assets/videos/ai_math1.mp4"
segment_video = "outputs/ai_math1_segment_2m20s.mp4"
segment_h264 = "outputs/ai_math1_segment_2m20s_h264.mp4"

print("\n📹 Step 1: Extracting segment from original video")
print(f"  Source: {input_video}")
print(f"  Time range: 2:20 to 2:24 (140-144 seconds)")
print(f"  Output: {segment_video}")

# Extract segment using ffmpeg (from 140s to 144s)
extract_cmd = [
    "ffmpeg", "-i", input_video,
    "-ss", "140",  # Start at 2:20 (140 seconds)
    "-t", "4",     # Duration 4 seconds
    "-c:v", "copy", "-c:a", "copy",
    "-y", segment_video
]

print("\nExtracting segment...")
result = subprocess.run(extract_cmd, capture_output=True, text=True)

if result.returncode != 0:
    print(f"❌ Failed to extract segment: {result.stderr}")
    sys.exit(1)

print("✅ Segment extracted successfully!")

# Convert to H.264 for compatibility
convert_cmd = [
    "ffmpeg", "-i", segment_video,
    "-c:v", "libx264", "-preset", "fast", "-crf", "23",
    "-pix_fmt", "yuv420p", "-movflags", "+faststart",
    "-y", segment_h264
]

print("\nConverting to H.264 for compatibility...")
result = subprocess.run(convert_cmd, capture_output=True, text=True)

if result.returncode == 0:
    print(f"✅ H.264 version created: {segment_h264}")
else:
    print("⚠️ H.264 conversion failed, using original segment")
    segment_h264 = segment_video

# Step 2: Apply Hello World animation using the refactored module
animated_video = "outputs/hello_world_segment_2m20s.mp4"
animated_h264 = "outputs/hello_world_segment_2m20s_h264.mp4"

print("\n📝 Step 2: Applying 'Hello World' animation")
print(f"  Input: {segment_h264}")
print(f"  Text: 'Hello World'")
print(f"  Position: Center (640, 400)")
print(f"  Output: {animated_h264}")

# Run animation with refactored module
animation_cmd = [
    "python", "-m", "utils.animations.apply_3d_text_animation",
    segment_h264,
    "--text", "Hello World",
    "--position", "640,400",  # Center position
    "--motion-duration", "0.8",
    "--dissolve-duration", "2.5",
    "--output", animated_video,
    "--supersample", "8",
]

print("\nApplying text animation with refactored module...")
print("-"*50)

result = subprocess.run(animation_cmd, capture_output=True, text=True)

if result.returncode == 0:
    print("✅ Animation applied successfully!")
    
    # Get the actual output filename (might have _hq suffix)
    if not os.path.exists(animated_video) and os.path.exists(animated_video.replace('.mp4', '_hq.mp4')):
        animated_video = animated_video.replace('.mp4', '_hq.mp4')
    
    # Convert to H.264
    print("\nConverting final video to H.264...")
    final_cmd = [
        "ffmpeg", "-i", animated_video,
        "-c:v", "libx264", "-preset", "slow", "-crf", "18",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        "-y", animated_h264
    ]
    
    result = subprocess.run(final_cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"✅ Final H.264 video created: {animated_h264}")
        
        # Get file info
        size_mb = os.path.getsize(animated_h264) / (1024 * 1024)
        
        print("\n" + "="*80)
        print("✅ SUCCESS!")
        print("="*80)
        print(f"\n📹 Final Video Details:")
        print(f"  File: {animated_h264}")
        print(f"  Size: {size_mb:.2f} MB")
        print(f"  Segment: 2:20-2:24 from original video")
        print(f"  Text: 'Hello World' with 3D effects")
        
        print("\n✨ Animation Features:")
        print("  • Motion phase (0.8s) - 3D text emergence")
        print("  • Dissolve phase (2.5s) - Letter-by-letter fade")
        print("  • Dynamic occlusion - Text behind speaker")
        print("  • High quality - 8x supersampling")
        
        print("\n🔧 Using Refactored Module:")
        print("  utils/animations/letter_3d_dissolve/")
        print("  • dissolve.py - Main animation class")
        print("  • timing.py - Frame-accurate scheduling")
        print("  • renderer.py - 3D letter rendering")
        print("  • occlusion.py - Dynamic masking")
        print("  • ...and 4 more modules")
        
        print("\n👀 View the result:")
        print(f"  {animated_h264}")
        
    else:
        print(f"❌ Final H.264 conversion failed")
else:
    print(f"❌ Animation failed")
    if result.stderr:
        print(f"Error: {result.stderr[:500]}")
    sys.exit(1)