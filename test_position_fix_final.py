#!/usr/bin/env python3
"""Test the corrected position fix - letters stay centered"""

import cv2
import numpy as np
from utils.animations.text_3d_motion_dissolve_correct import Text3DMotionDissolve

print("="*80)
print("TESTING CORRECTED POSITION FIX")
print("="*80)
print("\n✅ Fix applied:")
print("  • Letters positioned to maintain center at x=583")
print("  • No left shift at phase transition")
print("  • Continuous position from motion to dissolve")
print("")

# Load video
video_path = "test_element_3sec.mp4"
cap = cv2.VideoCapture(video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Load frames
frames = []
for i in range(136):
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = cap.read()
    frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
cap.release()

# Create animation with debug
motion_duration = 0.75
dissolve_duration = 1.5
total_duration = motion_duration + dissolve_duration

print("Creating animation with corrected positioning...")
anim = Text3DMotionDissolve(
    duration=total_duration,
    fps=fps,
    resolution=(W, H),
    text="HELLO WORLD",
    segment_mask=None,
    font_size=140,
    text_color=(255, 220, 0),
    depth_color=(200, 170, 0),
    depth_layers=8,
    depth_offset=3,
    motion_duration=motion_duration,
    start_scale=2.0,
    end_scale=1.0,
    final_scale=0.9,
    shrink_duration=0.6,
    settle_duration=0.15,
    dissolve_stable_duration=0.1,
    dissolve_duration=0.5,
    dissolve_stagger=0.1,
    float_distance=40,
    max_dissolve_scale=1.3,
    randomize_order=False,  # Consistent for testing
    maintain_kerning=True,
    center_position=(W//2, H//2),
    shadow_offset=6,
    outline_width=2,
    perspective_angle=0,
    supersample_factor=2,
    glow_effect=True,
    debug=True,  # Enable debug
)

print("\n" + "="*80)
print("GENERATING TEST FRAMES")
print("="*80)

# Generate frames around transition
transition_frame = anim.motion_frames
test_frames = [
    transition_frame - 3,
    transition_frame - 2,
    transition_frame - 1,
    transition_frame,
    transition_frame + 1,
    transition_frame + 2,
    transition_frame + 5,
]

generated = []
for i in test_frames:
    if i == transition_frame - 1:
        print(f"\n>>> Generating frame {i}: LAST MOTION")
    elif i == transition_frame:
        print(f"\n>>> Generating frame {i}: FIRST DISSOLVE")
    else:
        print(f"\nGenerating frame {i}")
    
    frame = anim.generate_frame(i, frames[i])
    if frame.shape[2] == 4:
        frame = frame[:, :, :3]
    generated.append((i, frame))

print("\n" + "="*80)
print("ANALYZING TEXT POSITIONS")
print("="*80)

def detect_text_center(frame_rgb):
    """Detect center of yellow text"""
    hsv = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2HSV)
    
    # Detect yellow
    lower = np.array([20, 100, 100])
    upper = np.array([40, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    
    # Find bounds
    pixels = np.where(mask > 0)
    if len(pixels[0]) > 0:
        min_x = np.min(pixels[1])
        max_x = np.max(pixels[1])
        center_x = (min_x + max_x) / 2
        return center_x, min_x, max_x
    return None, None, None

print("\nText positions:")
print("-" * 60)
print("Frame | Center X | Left X | Right X | Phase")
print("-" * 60)

positions = []
for idx, frame_rgb in generated:
    center_x, left_x, right_x = detect_text_center(frame_rgb)
    if center_x:
        phase = "MOTION" if idx < transition_frame else "DISSOLVE"
        marker = ""
        if idx == transition_frame - 1:
            marker = " <-- LAST MOTION"
        elif idx == transition_frame:
            marker = " <-- FIRST DISSOLVE"
        
        positions.append((idx, center_x, left_x, right_x))
        print(f"{idx:5d} | {center_x:8.1f} | {left_x:6.1f} | {right_x:7.1f} | {phase:8s}{marker}")

# Check for jump
if len(positions) >= 2:
    print("\n" + "="*80)
    print("POSITION CONTINUITY CHECK")
    print("="*80)
    
    # Find transition
    last_motion = None
    first_dissolve = None
    
    for idx, cx, lx, rx in positions:
        if idx == transition_frame - 1:
            last_motion = (cx, lx, rx)
        elif idx == transition_frame:
            first_dissolve = (cx, lx, rx)
    
    if last_motion and first_dissolve:
        center_shift = first_dissolve[0] - last_motion[0]
        left_shift = first_dissolve[1] - last_motion[1]
        
        print(f"\nTransition analysis:")
        print(f"  Last motion center: {last_motion[0]:.1f}")
        print(f"  First dissolve center: {first_dissolve[0]:.1f}")
        print(f"  Center shift: {center_shift:.1f} pixels")
        print(f"  Left edge shift: {left_shift:.1f} pixels")
        
        if abs(center_shift) < 5 and abs(left_shift) < 5:
            print("\n✅ POSITION CONTINUITY: EXCELLENT!")
            print("   No significant shift detected")
        else:
            print(f"\n⚠️ Position shift detected: {center_shift:.1f} pixels")

# Save comparison frames
print("\n" + "="*80)
print("SAVING VERIFICATION FRAMES")
print("="*80)

for idx, frame_rgb in generated:
    if idx == transition_frame - 1:
        cv2.imwrite('corrected_last_motion.png', cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
        print(f"Saved: corrected_last_motion.png")
    elif idx == transition_frame:
        cv2.imwrite('corrected_first_dissolve.png', cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
        print(f"Saved: corrected_first_dissolve.png")

# Generate full video
print("\n" + "="*80)
print("GENERATING FULL VIDEO")
print("="*80)

print("Generating all frames...")
output_frames = []
for i in range(min(len(frames), 136)):
    if i % 20 == 0:
        print(f"  Frame {i}/136...")
    frame = anim.generate_frame(i, frames[i])
    if frame.shape[2] == 4:
        frame = frame[:, :, :3]
    output_frames.append(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

print("\nSaving video...")
out = cv2.VideoWriter("corrected_animation.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (W, H))
for f in output_frames:
    out.write(f)
out.release()

# Convert to H.264
import subprocess
print("Converting to H.264...")
subprocess.run([
    'ffmpeg', '-y', '-i', 'corrected_animation.mp4',
    '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
    '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
    'CORRECTED_3D_ANIMATION_h264.mp4'
], capture_output=True)

import os
os.remove('corrected_animation.mp4')

print("\n✅ TEST COMPLETE!")
print("\n📹 Video: CORRECTED_3D_ANIMATION_h264.mp4")
print("\nThe text should now maintain its center position throughout the transition!")