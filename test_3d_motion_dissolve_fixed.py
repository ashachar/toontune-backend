#!/usr/bin/env python3
"""Test the fixed 3D text animation with per-letter position tracking."""

import cv2
import numpy as np
from utils.animations.text_3d_motion_dissolve_fixed import Text3DMotionDissolveFixed

print("="*80)
print("TESTING FIXED 3D TEXT ANIMATION")
print("="*80)

# Load test video
video_path = "test_element_3sec.mp4"
cap = cv2.VideoCapture(video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Read background frames
frames = []
for i in range(90):  # 3 seconds at 30fps
    ret, frame = cap.read()
    if not ret:
        break
    frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
cap.release()

print(f"\nLoaded {len(frames)} frames")
print(f"Resolution: {W}x{H}")
print(f"FPS: {fps}")

# Create animation with debug enabled
print("\nCreating fixed animation...")
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

# Generate video with the fixed animation
print("\nGenerating video...")
output_path = "text_3d_motion_dissolve_FIXED.mp4"
anim.generate_video(output_path, frames)

# Convert to H.264
print("\nConverting to H.264...")
h264_path = "text_3d_motion_dissolve_FIXED_h264.mp4"
import subprocess
subprocess.run([
    'ffmpeg', '-i', output_path,
    '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
    '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
    h264_path, '-y'
], check=True)

print(f"\n✅ Fixed animation saved to: {h264_path}")

# Generate frame comparison at transition
print("\n" + "="*80)
print("GENERATING TRANSITION FRAME COMPARISON")
print("="*80)

transition_frame = anim.motion_frames
print(f"\nTransition at frame {transition_frame}")

# Generate last motion and first dissolve frames
last_motion = anim.generate_frame(transition_frame - 1, frames[transition_frame - 1])
first_dissolve = anim.generate_frame(transition_frame, frames[transition_frame])

# Remove alpha channel if present
if last_motion.shape[2] == 4:
    last_motion = last_motion[:, :, :3]
if first_dissolve.shape[2] == 4:
    first_dissolve = first_dissolve[:, :, :3]

# Create side-by-side comparison
comparison = np.hstack([last_motion, first_dissolve])

# Add labels
from PIL import Image, ImageDraw, ImageFont
img = Image.fromarray(comparison)
draw = ImageDraw.Draw(img)

# Add labels
try:
    font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 20)
except:
    font = ImageFont.load_default()

draw.text((10, 10), "Last Motion Frame", fill=(255, 255, 255), font=font)
draw.text((W + 10, 10), "First Dissolve Frame", fill=(255, 255, 255), font=font)

# Draw center lines
center_x = W // 2
draw.line([(center_x, 0), (center_x, H)], fill=(0, 255, 0), width=1)
draw.line([(W + center_x, 0), (W + center_x, H)], fill=(0, 255, 0), width=1)

# Save comparison
comparison_array = np.array(img)
cv2.imwrite('transition_comparison_FIXED.png', cv2.cvtColor(comparison_array, cv2.COLOR_RGB2BGR))
print("Saved: transition_comparison_FIXED.png")

# Calculate difference
diff = cv2.absdiff(last_motion, first_dissolve)
diff_gray = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
_, diff_thresh = cv2.threshold(diff_gray, 5, 255, cv2.THRESH_BINARY)

# Count moving pixels
moving_pixels = np.sum(diff_thresh > 0)
total_pixels = diff_thresh.shape[0] * diff_thresh.shape[1]
movement_percentage = (moving_pixels / total_pixels) * 100

print(f"\nMovement analysis:")
print(f"  Moving pixels: {moving_pixels:,}")
print(f"  Total pixels: {total_pixels:,}") 
print(f"  Movement: {movement_percentage:.2f}%")

if movement_percentage < 2:
    print("\n✅ EXCELLENT! Individual letters maintain position perfectly!")
elif movement_percentage < 5:
    print("\n✅ GOOD! Minimal movement detected - letters are stable!")
else:
    print(f"\n⚠️ Still detecting movement: {movement_percentage:.1f}% of pixels changed")

print("\n✅ TEST COMPLETE")
print(f"\nCheck the video: {h264_path}")
print(f"Check the comparison: transition_comparison_FIXED.png")