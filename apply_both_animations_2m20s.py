#!/usr/bin/env python3
"""
Apply BOTH motion and dissolve animations to the 2:20-2:24 segment.
First the 3D text motion animation, then handoff to dissolve.
"""

import cv2
import numpy as np
import os
import sys

print("="*80)
print("🎬 APPLYING MOTION + DISSOLVE TO 2:20-2:24 SEGMENT")
print("="*80)

# Configuration
INPUT_VIDEO = "uploads/assets/videos/ai_math1.mp4"
SEGMENT_VIDEO = "outputs/ai_math1_segment_2m20s.mp4"
OUTPUT_VIDEO = "outputs/hello_world_both_animations_2m20s.mp4"

# Extract segment if not already done
if not os.path.exists(SEGMENT_VIDEO):
    print("\n📹 Extracting 2:20-2:24 segment...")
    extract_cmd = (
        f'ffmpeg -i {INPUT_VIDEO} -ss 140 -t 4 '
        f'-c:v copy -c:a copy {SEGMENT_VIDEO} -y'
    )
    os.system(extract_cmd + ' 2>/dev/null')
else:
    print(f"\n✅ Using existing segment: {SEGMENT_VIDEO}")

# Get video properties
cap = cv2.VideoCapture(SEGMENT_VIDEO)
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
cap.release()

print(f"  Video: {width}x{height}, {frame_count} frames @ {fps} fps")

# Import both animation modules
print("\n🎨 Loading animation modules...")
from utils.animations.text_3d_motion import Text3DMotion
from utils.animations.letter_3d_dissolve import Letter3DDissolve

# Animation timing
MOTION_DURATION = 0.8  # seconds
SAFETY_HOLD = 0.5      # seconds
DISSOLVE_START = MOTION_DURATION + SAFETY_HOLD  # 1.3 seconds

motion_end_frame = int(MOTION_DURATION * fps)
dissolve_start_frame = int(DISSOLVE_START * fps)

print(f"\n📊 Animation timeline:")
print(f"  Motion: frames 0-{motion_end_frame} (0.0-{MOTION_DURATION}s)")
print(f"  Hold: frames {motion_end_frame}-{dissolve_start_frame} ({MOTION_DURATION}-{DISSOLVE_START}s)")
print(f"  Dissolve: frames {dissolve_start_frame}-{frame_count} ({DISSOLVE_START}s onwards)")

# Create motion animation
print("\n🎬 Creating motion animation...")
motion_animation = Text3DMotion(
    text="Hello World",
    position=(width//2, height//2 - 50),
    font_size=72,
    duration=MOTION_DURATION,
    fps=fps,
    resolution=(width, height),
    is_behind=True,
    start_scale=0.3,
    end_scale=1.0,
    supersample_factor=8,
    debug=False
)

print("🎬 Creating dissolve animation...")
# For dissolve, we need to get the handoff state from motion
# The dissolve should start from where motion ended

# Process frames
print(f"\n📹 Processing {frame_count} frames...")
cap = cv2.VideoCapture(SEGMENT_VIDEO)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))

handoff_state = None
dissolve_animation = None

for frame_idx in range(frame_count):
    ret, frame = cap.read()
    if not ret:
        break
    
    if frame_idx < motion_end_frame:
        # MOTION PHASE
        phase = "MOTION"
        # Apply motion animation
        frame_with_text = motion_animation.generate_frame(frame_idx, frame)
        
        # Capture handoff state at the end of motion
        if frame_idx == motion_end_frame - 1:
            handoff_state = motion_animation.get_handoff_state()
            print(f"\n📦 Captured handoff state at frame {frame_idx}")
            
    elif frame_idx < dissolve_start_frame:
        # SAFETY HOLD PHASE - show the final frame from motion
        phase = "HOLD"
        # Keep showing the last motion frame
        frame_with_text = motion_animation.generate_frame(motion_end_frame - 1, frame)
        
        # Create dissolve animation if not yet created
        if dissolve_animation is None and handoff_state is not None:
            print(f"\n🔄 Creating dissolve animation from handoff state...")
            
            # Create dissolve with handoff parameters
            from utils.animations.letter_3d_dissolve.handoff import HandoffHandler
            
            # Initialize dissolve animation
            dissolve_animation = Letter3DDissolve(
                text="Hello World",
                initial_position=handoff_state.position if handoff_state else (width//2, height//2 - 50),
                initial_scale=handoff_state.scale if handoff_state else 1.0,
                is_behind=True,
                font_size=72,
                resolution=(width, height),
                fps=fps,
                duration=3.0,  # Dissolve duration
                stable_duration=0.0,  # No motion in dissolve
                dissolve_duration=2.5,
                supersample_factor=8,
                debug=False
            )
            
            # Set up handoff if we have letter sprites
            if handoff_state and hasattr(handoff_state, 'letter_sprites'):
                dissolve_animation.sprite_manager.letter_sprites = handoff_state.letter_sprites
                dissolve_animation.handoff_handler.motion_state = handoff_state
                print(f"  ✅ Transferred {len(handoff_state.letter_sprites)} letter sprites")
    else:
        # DISSOLVE PHASE
        phase = "DISSOLVE"
        if dissolve_animation:
            # Adjust frame index for dissolve (relative to dissolve start)
            dissolve_frame = frame_idx - dissolve_start_frame
            frame_with_text = dissolve_animation.generate_frame(dissolve_frame, frame)
        else:
            frame_with_text = frame
    
    # Ensure numpy array
    if not isinstance(frame_with_text, np.ndarray):
        frame_with_text = np.array(frame_with_text)
    
    # Ensure BGR format
    if len(frame_with_text.shape) == 3 and frame_with_text.shape[2] == 3:
        # Check if it needs BGR conversion (assuming generate_frame returns BGR)
        pass
    
    # Add phase indicator
    cv2.putText(frame_with_text, f"{phase} | 2:{20+int(frame_idx/fps):02d}",
               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
               (0, 255, 0) if phase == "HOLD" else (255, 255, 255), 2)
    
    out.write(frame_with_text)
    
    if frame_idx % 25 == 0:
        print(f"  Frame {frame_idx}/{frame_count} - {phase}")

cap.release()
out.release()

print(f"\n✅ Animation complete: {OUTPUT_VIDEO}")

# Convert to H.264
print("\n🎬 Converting to H.264...")
h264_output = OUTPUT_VIDEO.replace('.mp4', '_h264.mp4')
convert_cmd = (
    f'ffmpeg -i {OUTPUT_VIDEO} -c:v libx264 -preset fast -crf 23 '
    f'-pix_fmt yuv420p -movflags +faststart {h264_output} -y'
)
os.system(convert_cmd + ' 2>/dev/null')

if os.path.exists(h264_output):
    size_mb = os.path.getsize(h264_output) / (1024 * 1024)
    print(f"✅ H.264 version: {h264_output} ({size_mb:.2f} MB)")
    print(f"\n🎥 Opening video...")
    os.system(f"open {h264_output}")
else:
    print(f"⚠️ H.264 conversion failed, opening original...")
    os.system(f"open {OUTPUT_VIDEO}")

print("\n" + "="*80)
print("✅ COMPLETE: Motion + Dissolve animation on 2:20-2:24")
print("="*80)
print("\n🎯 Animation phases:")
print(f"  1. MOTION (0.0-{MOTION_DURATION}s): 3D text emerges and moves")
print(f"  2. HOLD ({MOTION_DURATION}-{DISSOLVE_START}s): Text stays stable")
print(f"  3. DISSOLVE ({DISSOLVE_START}s+): Letter-by-letter dissolve")
print("="*80)