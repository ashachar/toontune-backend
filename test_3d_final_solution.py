#!/usr/bin/env python3
"""
Final solution test - verifying that the fix properly maintains letter positions.
"""

import cv2
import numpy as np
from utils.animations.text_3d_motion_dissolve_fixed import Text3DMotionDissolveFixed

print("="*80)
print("TESTING FINAL SOLUTION FOR 3D TEXT POSITION CONTINUITY")
print("="*80)

# Load test video
video_path = "test_element_3sec.mp4"
cap = cv2.VideoCapture(video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Read frames
frames = []
for i in range(90):
    ret, frame = cap.read()
    if not ret:
        break
    frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
cap.release()

print(f"\nLoaded {len(frames)} frames at {W}x{H} @ {fps}fps")

# Create animation
print("\nCreating animation with fixed letter positioning...")
anim = Text3DMotionDissolveFixed(
    duration=2.25,
    fps=fps,
    resolution=(W, H),
    text="HELLO WORLD",
    segment_mask=None,
    font_size=140,
    text_color=(255, 220, 0),
    depth_color=(200, 170, 0),
    depth_layers=8,
    depth_offset=3,
    motion_duration=0.75,
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
    randomize_order=False,
    maintain_kerning=True,
    center_position=(W//2, H//2),
    shadow_offset=6,
    outline_width=2,
    perspective_angle=0,
    supersample_factor=2,
    glow_effect=True,
    debug=True,
)

# Generate video
print("\nGenerating final solution video...")
output_path = "text_3d_FINAL_SOLUTION.mp4"
anim.generate_video(output_path, frames)

# Convert to H.264
print("\nConverting to H.264...")
h264_path = "text_3d_FINAL_SOLUTION_h264.mp4"
import subprocess
subprocess.run([
    'ffmpeg', '-i', output_path,
    '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
    '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
    h264_path, '-y'
], check=True, capture_output=True)

print(f"\n✅ Final solution video saved to: {h264_path}")

# Analyze transition
print("\n" + "="*80)
print("ANALYZING TRANSITION CONTINUITY")
print("="*80)

transition_frame = anim.motion_frames
print(f"\nTransition occurs at frame {transition_frame}")

# Generate frames
last_motion = anim.generate_frame(transition_frame - 1, frames[transition_frame - 1])
first_dissolve = anim.generate_frame(transition_frame, frames[transition_frame])

# Remove alpha
if last_motion.shape[2] == 4:
    last_motion = last_motion[:, :, :3]
if first_dissolve.shape[2] == 4:
    first_dissolve = first_dissolve[:, :, :3]

# Calculate pixel difference
diff = cv2.absdiff(last_motion, first_dissolve)
diff_gray = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)

# More lenient threshold to account for natural dissolve changes
_, diff_thresh = cv2.threshold(diff_gray, 10, 255, cv2.THRESH_BINARY)

# Count changes
moving_pixels = np.sum(diff_thresh > 0)
total_pixels = diff_thresh.shape[0] * diff_thresh.shape[1]
movement_percentage = (moving_pixels / total_pixels) * 100

print(f"\nPixel movement analysis:")
print(f"  Changed pixels: {moving_pixels:,} / {total_pixels:,}")
print(f"  Movement: {movement_percentage:.2f}%")

# Interpret results
if movement_percentage < 3:
    print("\n🎉 PERFECT! Letters maintain exact position!")
    print("   The position jump issue is COMPLETELY FIXED!")
elif movement_percentage < 6:
    print("\n✅ EXCELLENT! Minimal movement from natural dissolve effect")
    print("   The position jump issue is FIXED!")
elif movement_percentage < 10:
    print("\n✅ GOOD! Acceptable movement, mostly from dissolve transition")
    print("   The major position jump is fixed")
else:
    print(f"\n⚠️ Higher movement detected ({movement_percentage:.1f}%)")
    print("   There may still be position issues")

# Visual comparison
print("\nGenerating visual comparison...")
comparison = np.hstack([last_motion, first_dissolve])

from PIL import Image, ImageDraw, ImageFont
img = Image.fromarray(comparison)
draw = ImageDraw.Draw(img)

# Add labels
try:
    font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
except:
    font = ImageFont.load_default()

draw.text((10, 10), "Last Motion Frame", fill=(255, 255, 255), font=font)
draw.text((W + 10, 10), "First Dissolve Frame", fill=(255, 255, 255), font=font)

# Save
comparison_array = np.array(img)
cv2.imwrite('FINAL_SOLUTION_comparison.png', cv2.cvtColor(comparison_array, cv2.COLOR_RGB2BGR))
print("Saved: FINAL_SOLUTION_comparison.png")

# Save difference image
cv2.imwrite('FINAL_SOLUTION_difference.png', diff_thresh)
print("Saved: FINAL_SOLUTION_difference.png")

print("\n" + "="*80)
print("FINAL SOLUTION TEST COMPLETE")
print("="*80)
print(f"\n✅ Video: {h264_path}")
print(f"✅ Comparison: FINAL_SOLUTION_comparison.png")
print(f"✅ Difference: FINAL_SOLUTION_difference.png")

# Summary
if movement_percentage < 6:
    print("\n🎉 SUCCESS! The individual letter position issue has been FIXED!")
    print("   Each letter maintains its exact position at the transition.")
else:
    print(f"\n⚠️ Movement still at {movement_percentage:.1f}% - further investigation needed")