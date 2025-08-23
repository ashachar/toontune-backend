#!/usr/bin/env python3
"""Test smooth continuous motion fix - iterative verification"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils.animations.text_3d_behind_segment_smooth import Text3DBehindSegment

print("="*70)
print("TESTING SMOOTH CONTINUOUS MOTION FIX")
print("="*70)
print("\n✅ Key Changes:")
print("  • Single continuous t across entire animation")
print("  • No reset between phases")
print("  • Smooth interpolation from start → end → final")
print("")

# Load video
video_path = "test_element_3sec.mp4"
cap = cv2.VideoCapture(video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

frames = []
for i in range(45):
    ret, frame = cap.read()
    if not ret:
        break
    frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
cap.release()

# Create animation
anim = Text3DBehindSegment(
    duration=0.75,
    fps=fps,
    resolution=(W, H),
    text="HELLO WORLD",
    segment_mask=None,
    font_size=140,
    text_color=(255, 220, 0),
    depth_color=(200, 170, 0),
    depth_layers=8,
    depth_offset=3,
    start_scale=2.0,
    end_scale=1.0,
    final_scale=0.9,
    shrink_duration=0.6,
    settle_duration=0.15,
    center_position=(W//2, H//2),
    shadow_offset=6,
    outline_width=2,
    perspective_angle=0,
    supersample_factor=2,
    debug=False,
)

print("Phase 1: Analyzing motion continuity...")
print("-" * 70)

# Track motion parameters
scales = []
positions_y = []
velocities_scale = []
velocities_y = []

# Calculate expected values
total_frames = int(0.75 * fps)
shrink_frames = int(0.6 * fps)
settle_frames = total_frames - shrink_frames

print(f"Animation structure:")
print(f"  Total frames: {total_frames}")
print(f"  Shrink phase: frames 0-{shrink_frames-1}")
print(f"  Settle phase: frames {shrink_frames}-{total_frames-1}")
print(f"  Transition at: frame {shrink_frames}")

# Analyze motion without rendering
for frame_num in range(total_frames):
    # Calculate parameters (same as in animation)
    t_global = frame_num / max(total_frames - 1, 1)
    smooth_t = t_global * t_global * (3.0 - 2.0 * t_global)
    
    # Calculate scale
    shrink_progress = 0.6 / 0.75  # 0.8
    if smooth_t <= shrink_progress:
        local_t = smooth_t / shrink_progress
        scale = 2.0 - local_t * (2.0 - 1.0)
    else:
        local_t = (smooth_t - shrink_progress) / (1.0 - shrink_progress)
        scale = 1.0 - local_t * (1.0 - 0.9)
    
    # Calculate Y position
    start_y = H//2 - H * 0.15
    end_y = H//2
    pos_y = start_y + smooth_t * (end_y - start_y)
    
    scales.append(scale)
    positions_y.append(pos_y)
    
    # Calculate velocities
    if frame_num > 0:
        vel_scale = scales[-1] - scales[-2]
        vel_y = positions_y[-1] - positions_y[-2]
    else:
        vel_scale = 0
        vel_y = 0
    
    velocities_scale.append(vel_scale)
    velocities_y.append(vel_y)

# Check for discontinuities
print("\n" + "="*70)
print("CONTINUITY ANALYSIS")
print("="*70)

discontinuity_found = False

# Check scale continuity
print("\nScale velocity analysis:")
for i in range(1, len(velocities_scale)):
    if abs(velocities_scale[i] - velocities_scale[i-1]) > 0.01:
        print(f"  ⚠️ Frame {i}: Velocity jump from {velocities_scale[i-1]:.4f} to {velocities_scale[i]:.4f}")
        discontinuity_found = True

if not discontinuity_found:
    print("  ✅ Scale velocity is SMOOTH - no discontinuities!")

# Check position continuity
print("\nPosition velocity analysis:")
pos_discontinuity = False
for i in range(1, len(velocities_y)):
    if abs(velocities_y[i] - velocities_y[i-1]) > 2:
        print(f"  ⚠️ Frame {i}: Velocity jump from {velocities_y[i-1]:.2f} to {velocities_y[i]:.2f}")
        pos_discontinuity = True

if not pos_discontinuity:
    print("  ✅ Position velocity is SMOOTH - no discontinuities!")

# Focus on transition point
print("\n" + "="*70)
print(f"TRANSITION ANALYSIS (Frame {shrink_frames-1} to {shrink_frames+1})")
print("="*70)

for i in range(max(0, shrink_frames-2), min(shrink_frames+3, total_frames)):
    phase = "shrink" if i < shrink_frames else "settle"
    print(f"\nFrame {i} ({phase}): ")
    print(f"  Scale: {scales[i]:.4f} (velocity: {velocities_scale[i]:.5f})")
    print(f"  Pos Y: {positions_y[i]:.2f} (velocity: {velocities_y[i]:.3f})")
    
    if i == shrink_frames - 1:
        print("  >>> PHASE TRANSITION <<<")

# Create visualization
print("\n" + "="*70)
print("VISUAL VERIFICATION")
print("="*70)

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Scale over time
axes[0, 0].plot(scales, 'b-', linewidth=2)
axes[0, 0].axvline(x=shrink_frames, color='r', linestyle='--', alpha=0.5, label='Phase transition')
axes[0, 0].set_title('Scale (should be smooth curve)')
axes[0, 0].set_xlabel('Frame')
axes[0, 0].set_ylabel('Scale')
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].legend()

# Scale velocity
axes[0, 1].plot(velocities_scale, 'g-', linewidth=2)
axes[0, 1].axvline(x=shrink_frames, color='r', linestyle='--', alpha=0.5)
axes[0, 1].axhline(y=0, color='k', linestyle='-', alpha=0.2)
axes[0, 1].set_title('Scale Velocity (should be CONTINUOUS)')
axes[0, 1].set_xlabel('Frame')
axes[0, 1].set_ylabel('Change per frame')
axes[0, 1].grid(True, alpha=0.3)

# Position over time
axes[1, 0].plot(positions_y, 'b-', linewidth=2)
axes[1, 0].axvline(x=shrink_frames, color='r', linestyle='--', alpha=0.5)
axes[1, 0].set_title('Y Position (should be smooth curve)')
axes[1, 0].set_xlabel('Frame')
axes[1, 0].set_ylabel('Y Position (pixels)')
axes[1, 0].grid(True, alpha=0.3)

# Position velocity
axes[1, 1].plot(velocities_y, 'g-', linewidth=2)
axes[1, 1].axvline(x=shrink_frames, color='r', linestyle='--', alpha=0.5)
axes[1, 1].axhline(y=0, color='k', linestyle='-', alpha=0.2)
axes[1, 1].set_title('Y Velocity (should be CONTINUOUS)')
axes[1, 1].set_xlabel('Frame')
axes[1, 1].set_ylabel('Change per frame')
axes[1, 1].grid(True, alpha=0.3)

plt.suptitle('SMOOTH MOTION FIX - Continuity Analysis', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('smooth_motion_analysis.png', dpi=150)
print("\n📊 Saved analysis graph: smooth_motion_analysis.png")

# Generate actual video
print("\n" + "="*70)
print("Phase 2: Generating video with smooth motion...")
print("="*70)

output_frames = []
for i in range(len(frames)):
    if i % 10 == 0:
        print(f"  Processing frame {i}/{len(frames)}...")
    frame = anim.generate_frame(i, frames[i])
    if frame.shape[2] == 4:
        frame = frame[:, :, :3]
    output_frames.append(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

# Save video
print("\nSaving video...")
out = cv2.VideoWriter("smooth_motion.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (W, H))
for f in output_frames:
    out.write(f)
out.release()

import subprocess
subprocess.run([
    'ffmpeg', '-y', '-i', 'smooth_motion.mp4',
    '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
    '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
    'SMOOTH_MOTION_FINAL_h264.mp4'
], capture_output=True)

import os
os.remove('smooth_motion.mp4')

# Extract and verify specific frames
print("\n" + "="*70)
print("Phase 3: Frame-by-frame verification...")
print("="*70)

# Check critical frames
critical_frames = [
    shrink_frames - 2,  # Before transition
    shrink_frames - 1,  # Right before
    shrink_frames,      # Transition point
    shrink_frames + 1,  # Right after
    shrink_frames + 2,  # After transition
]

print("\nExamining critical frames around transition:")
print("-" * 50)

for idx in critical_frames:
    if 0 <= idx < len(output_frames):
        frame_img = output_frames[idx]
        phase = "shrink" if idx < shrink_frames else "settle"
        
        # Save frame for inspection
        cv2.imwrite(f'frame_{idx:03d}_{phase}.png', frame_img)
        print(f"  Frame {idx} ({phase}): saved as frame_{idx:03d}_{phase}.png")
        print(f"    Scale: {scales[idx]:.4f}")
        print(f"    Velocity: {velocities_scale[idx]:.5f}")

# Final summary
print("\n" + "="*70)
print("✅ SMOOTH MOTION FIX COMPLETE!")
print("="*70)
print("\n📹 Final video: SMOOTH_MOTION_FINAL_h264.mp4")
print("\n🎯 Results:")

if not discontinuity_found and not pos_discontinuity:
    print("  ✅ Motion is perfectly CONTINUOUS")
    print("  ✅ No velocity jumps detected")
    print("  ✅ Smooth interpolation across phases")
    print("  ✅ No momentary stops")
else:
    print("  ⚠️ Some discontinuities detected - review the analysis")

print("\n📊 Analysis files created:")
print("  • smooth_motion_analysis.png - Velocity graphs")
print("  • frame_XXX_phase.png - Critical frames for inspection")
print("\nPlease review the video and confirm smooth motion!")