#!/usr/bin/env python3
"""
Apply the object-based "Hello World" animation to the 2:20-2:24 segment of ai_math1.mp4.
This demonstrates the fix for the stale mask bug using the new architecture.
"""

import cv2
import numpy as np
from PIL import Image
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.animations.object_system import (
    LetterObject,
    RenderPipeline
)
from utils.animations.object_system.animation_adapter import (
    MotionAnimationAdapter,
    DissolveAnimationAdapter
)
from utils.animations.letter_3d_dissolve.renderer import Letter3DRenderer

print("="*80)
print("🎬 APPLYING OBJECT-BASED ANIMATION TO 2:20-2:24 SEGMENT")
print("="*80)

# Configuration
INPUT_VIDEO = "uploads/assets/videos/ai_math1.mp4"
OUTPUT_VIDEO = "outputs/hello_world_object_system_2m20s.mp4"
START_TIME = 140.0  # 2:20 in seconds
DURATION = 4.0      # 4 seconds
TEXT = "Hello World"
FONT_SIZE = 72

# Timing configuration (in frames @ 25fps)
FPS = 25
MOTION_DURATION = 20    # 0.8 seconds
SAFETY_HOLD = 12        # 0.5 seconds  
DISSOLVE_DURATION = 62  # 2.5 seconds

print(f"\n📋 Configuration:")
print(f"  Input: {INPUT_VIDEO}")
print(f"  Segment: 2:20-2:24 (frames {int(START_TIME*FPS)}-{int((START_TIME+DURATION)*FPS)})")
print(f"  Text: '{TEXT}'")
print(f"  Output: {OUTPUT_VIDEO}")

# Open video and seek to start time
cap = cv2.VideoCapture(INPUT_VIDEO)
if not cap.isOpened():
    print(f"❌ Error: Cannot open {INPUT_VIDEO}")
    sys.exit(1)

# Get video properties
fps_actual = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Seek to start frame
start_frame = int(START_TIME * fps_actual)
cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

print(f"\n📹 Video properties:")
print(f"  Resolution: {width}x{height}")
print(f"  FPS: {fps_actual:.2f}")
print(f"  Starting at frame: {start_frame}")

# Initialize components
print("\n🔧 Initializing object-based render pipeline...")
pipeline = RenderPipeline(width, height, debug=False)
renderer = Letter3DRenderer(
    font_size=FONT_SIZE,
    supersample_factor=8,
    depth_layers=8,
    depth_offset=3
)

motion_adapter = MotionAnimationAdapter(
    motion_duration_frames=MOTION_DURATION,
    debug=False
)

dissolve_adapter = DissolveAnimationAdapter(
    motion_duration_frames=MOTION_DURATION,
    safety_hold_frames=SAFETY_HOLD,
    dissolve_duration_frames=DISSOLVE_DURATION,
    debug=False
)

# Create letter objects
print("\n📝 Creating letter objects...")
letters = []

# Calculate total width needed
total_width = 0
temp_sprites = []
for char in TEXT:
    if char == ' ':
        temp_sprites.append(None)
        total_width += FONT_SIZE // 2
    else:
        sprite_3d, anchor = renderer.render_3d_letter(char, 1.0, 1.0, 1.0)
        temp_sprites.append((char, sprite_3d, anchor))
        total_width += int(sprite_3d.width * 1.1)

# Center the text
start_x = (width - total_width) // 2
start_y = height // 2 - 50  # Slightly above center

# Create letter objects
current_x = start_x
for item in temp_sprites:
    if item is None:
        # Space
        current_x += FONT_SIZE // 2
        continue
    
    char, sprite_3d, anchor = item
    
    # Create letter object
    letter = LetterObject(
        char=char,
        position=(current_x, start_y),
        sprite_3d=sprite_3d,
        width=sprite_3d.width,
        height=sprite_3d.height,
        anchor=anchor
    )
    
    # CRITICAL: Set to be behind foreground for proper occlusion
    letter.set_behind(True)
    letter.base_position = (current_x, start_y)
    
    pipeline.add_object(letter)
    letters.append(letter)
    
    print(f"  '{char}' at ({current_x}, {start_y})")
    current_x += int(sprite_3d.width * 1.1)

print(f"\n✅ Created {len(letters)} letter objects with is_behind=True")

# Create video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps_out = int(fps_actual)
out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps_out, (width, height))

total_frames = int(DURATION * fps_actual)

print(f"\n🎥 Processing {total_frames} frames...")
print("  Motion phase: 0.0-0.8s")
print("  Safety hold: 0.8-1.3s")
print("  Dissolve phase: 1.3-3.8s")
print("  🔴 Watch occlusion update as speaker moves!")

# Process frames
for frame_idx in range(total_frames):
    ret, frame = cap.read()
    if not ret:
        print(f"⚠️ No more frames at {frame_idx}")
        break
    
    # Convert BGR to RGB for processing
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Determine animation phase and get transforms
    if frame_idx < MOTION_DURATION:
        phase = "MOTION"
        animations = motion_adapter.apply(letters, frame_idx, total_frames)
    elif frame_idx < MOTION_DURATION + SAFETY_HOLD:
        phase = "HOLD"
        animations = {}  # No animation during hold
    else:
        phase = "DISSOLVE"
        animations = dissolve_adapter.apply(letters, frame_idx, total_frames)
    
    # Progress indicator
    if frame_idx % 25 == 0:
        time_s = frame_idx / fps_actual
        print(f"  Frame {frame_idx}/{total_frames} (t={time_s:.1f}s) - {phase}")
    
    # Render frame with object system
    composite = pipeline.render_frame(
        background=frame_rgb,
        frame_number=frame_idx,
        animations=animations
    )
    
    # Convert back to BGR for video writing
    frame_bgr = cv2.cvtColor(composite, cv2.COLOR_RGB2BGR)
    
    # Add debug overlay
    cv2.putText(frame_bgr, f"{phase} | 2:{20+int(frame_idx/fps_actual):02d}",
               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
               (255, 255, 255), 2)
    
    out.write(frame_bgr)

# Cleanup
cap.release()
out.release()

print("\n" + "="*80)
print("✅ ANIMATION COMPLETE!")
print("="*80)
print(f"\n📹 Output saved to: {OUTPUT_VIDEO}")

# Convert to H.264 for compatibility
print("\n🎬 Converting to H.264 format...")
h264_output = OUTPUT_VIDEO.replace('.mp4', '_h264.mp4')
cmd = (f'ffmpeg -i {OUTPUT_VIDEO} -c:v libx264 -preset fast -crf 23 '
       f'-pix_fmt yuv420p -movflags +faststart {h264_output} -y')
os.system(cmd + ' 2>/dev/null')

if os.path.exists(h264_output):
    file_size = os.path.getsize(h264_output) / (1024 * 1024)
    print(f"✅ H.264 version created: {h264_output} ({file_size:.2f} MB)")
else:
    print("⚠️ H.264 conversion may have failed")

print("\n📊 Summary:")
print(f"  • Segment: 2:20-2:24 from ai_math1.mp4")
print(f"  • Text animated: '{TEXT}'")
print(f"  • Duration: {DURATION} seconds")
print(f"  • Resolution: {width}x{height}")
print(f"  • FPS: {fps_out}")

print("\n🎯 Key Features:")
print("  • Object-based architecture with state management")
print("  • Dynamic occlusion masks recalculated every frame")
print("  • Clean separation of animation and post-processing")
print("  • Fix for stale mask bug demonstrated")

print("\n💡 To view the result:")
print(f"  open {h264_output if os.path.exists(h264_output) else OUTPUT_VIDEO}")

print("="*80)