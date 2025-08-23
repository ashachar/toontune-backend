#!/usr/bin/env python3
"""Detect and analyze motion discontinuity in text animation"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils.animations.text_3d_behind_segment_no_grow import Text3DBehindSegment

print("="*70)
print("DETECTING MOTION DISCONTINUITY")
print("="*70)

# Set up animation parameters
fps = 60
W, H = 1166, 534
duration = 0.75
shrink_duration = 0.6
settle_duration = 0.15

# Create animation instance
anim = Text3DBehindSegment(
    duration=duration,
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
    shrink_duration=shrink_duration,
    settle_duration=settle_duration,
    center_position=(W//2, H//2),
    shadow_offset=6,
    outline_width=2,
    perspective_angle=0,
    supersample_factor=2,
    debug=False,
)

# Calculate frame counts
total_frames = int(duration * fps)
shrink_frames = int(shrink_duration * fps)
settle_frames = total_frames - shrink_frames

print(f"\nAnimation structure:")
print(f"  Total frames: {total_frames}")
print(f"  Shrink phase: frames 0-{shrink_frames-1}")
print(f"  Settle phase: frames {shrink_frames}-{total_frames-1}")
print(f"  Transition at: frame {shrink_frames}")

# Analyze motion parameters frame by frame
print("\n" + "="*70)
print("FRAME-BY-FRAME ANALYSIS")
print("="*70)

scales = []
positions_y = []
alphas = []
phases = []

# Track all parameters
for frame in range(total_frames):
    if frame < shrink_frames:
        phase = "shrink"
        t = frame / max(shrink_frames - 1, 1)
    else:
        phase = "settle"
        t = (frame - shrink_frames) / max(settle_frames - 1, 1)
    
    # Apply smoothstep
    smooth_t = t * t * (3.0 - 2.0 * t)
    
    # Calculate scale
    if phase == "shrink":
        scale = 2.0 - smooth_t * (2.0 - 1.0)
        
        # Calculate Y position
        start_y = H//2 - H * 0.15
        end_y = H//2
        pos_y = start_y + smooth_t * (end_y - start_y)
        
        # Calculate alpha
        if t < 0.4:
            alpha = 1.0
        elif t < 0.6:
            fade_t = (t - 0.4) / 0.2
            alpha = 1.0 - fade_t * 0.4
        else:
            fade_t = (t - 0.6) / 0.4
            k = 3.0
            alpha = 0.6 - 0.4 * (1 - np.exp(-k * fade_t)) / (1 - np.exp(-k))
    else:  # settle
        scale = 1.0 - smooth_t * (1.0 - 0.9)
        pos_y = H//2  # Should be constant
        alpha = 0.2
    
    scales.append(scale)
    positions_y.append(pos_y)
    alphas.append(alpha)
    phases.append(phase)

# Calculate velocities (change per frame)
scale_velocities = [0] + [scales[i] - scales[i-1] for i in range(1, len(scales))]
position_velocities = [0] + [positions_y[i] - positions_y[i-1] for i in range(1, len(positions_y))]

# Detect discontinuities
print("\nSearching for discontinuities...")
print("-" * 70)

# Check scale continuity
print("\nSCALE ANALYSIS:")
for i in range(1, len(scales)):
    if abs(scale_velocities[i] - scale_velocities[i-1]) > 0.01:  # Significant change in velocity
        print(f"  ⚠️ Frame {i}: Scale velocity jump from {scale_velocities[i-1]:.4f} to {scale_velocities[i]:.4f}")

# Check position continuity
print("\nPOSITION ANALYSIS:")
for i in range(1, len(positions_y)):
    if abs(position_velocities[i] - position_velocities[i-1]) > 2:  # Significant change in velocity
        print(f"  ⚠️ Frame {i}: Position velocity jump from {position_velocities[i-1]:.2f} to {position_velocities[i]:.2f}")

# Focus on transition point
print("\n" + "="*70)
print("TRANSITION POINT ANALYSIS (Frame 35-37)")
print("="*70)

for i in range(34, min(38, total_frames)):
    print(f"\nFrame {i} ({phases[i]} phase):")
    print(f"  Scale: {scales[i]:.4f} (velocity: {scale_velocities[i]:.4f})")
    print(f"  Pos Y: {positions_y[i]:.2f} (velocity: {position_velocities[i]:.2f})")
    print(f"  Alpha: {alphas[i]:.4f}")
    
    if i == shrink_frames - 1:
        print("  >>> PHASE TRANSITION <<<")

# Visualize the issue
print("\n" + "="*70)
print("VISUAL ANALYSIS")
print("="*70)

# Create plots
fig, axes = plt.subplots(3, 2, figsize=(12, 10))

# Scale over time
axes[0, 0].plot(scales, 'b-')
axes[0, 0].axvline(x=shrink_frames, color='r', linestyle='--', alpha=0.5)
axes[0, 0].set_title('Scale')
axes[0, 0].set_xlabel('Frame')
axes[0, 0].grid(True, alpha=0.3)

# Scale velocity
axes[0, 1].plot(scale_velocities, 'g-')
axes[0, 1].axvline(x=shrink_frames, color='r', linestyle='--', alpha=0.5)
axes[0, 1].set_title('Scale Velocity (should be smooth)')
axes[0, 1].set_xlabel('Frame')
axes[0, 1].grid(True, alpha=0.3)

# Position Y
axes[1, 0].plot(positions_y, 'b-')
axes[1, 0].axvline(x=shrink_frames, color='r', linestyle='--', alpha=0.5)
axes[1, 0].set_title('Y Position')
axes[1, 0].set_xlabel('Frame')
axes[1, 0].grid(True, alpha=0.3)

# Position velocity
axes[1, 1].plot(position_velocities, 'g-')
axes[1, 1].axvline(x=shrink_frames, color='r', linestyle='--', alpha=0.5)
axes[1, 1].set_title('Y Velocity (should be smooth)')
axes[1, 1].set_xlabel('Frame')
axes[1, 1].grid(True, alpha=0.3)

# Alpha
axes[2, 0].plot(alphas, 'b-')
axes[2, 0].axvline(x=shrink_frames, color='r', linestyle='--', alpha=0.5)
axes[2, 0].set_title('Alpha (transparency)')
axes[2, 0].set_xlabel('Frame')
axes[2, 0].grid(True, alpha=0.3)

# Combined view
axes[2, 1].plot(np.array(scales) / max(scales), 'b-', label='Scale (norm)', alpha=0.7)
axes[2, 1].plot(np.array(positions_y) / max(positions_y), 'g-', label='Pos Y (norm)', alpha=0.7)
axes[2, 1].plot(alphas, 'r-', label='Alpha', alpha=0.7)
axes[2, 1].axvline(x=shrink_frames, color='k', linestyle='--', alpha=0.5, label='Phase transition')
axes[2, 1].set_title('All parameters (normalized)')
axes[2, 1].set_xlabel('Frame')
axes[2, 1].legend()
axes[2, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('motion_discontinuity_analysis.png', dpi=150)
print("\nSaved analysis to: motion_discontinuity_analysis.png")

# Identify the problem
print("\n" + "="*70)
print("PROBLEM IDENTIFICATION")
print("="*70)

# Check for sudden stops
position_stopped = False
scale_stopped = False

for i in range(shrink_frames - 2, min(shrink_frames + 3, total_frames)):
    if abs(position_velocities[i]) < 0.1:
        position_stopped = True
        print(f"\n❌ Position STOPS at frame {i}")
    if abs(scale_velocities[i]) < 0.001:
        scale_stopped = True
        print(f"❌ Scale STOPS at frame {i}")

# Check for direction changes
for i in range(1, len(position_velocities)):
    if position_velocities[i-1] * position_velocities[i] < 0:  # Sign change
        print(f"\n❌ Position changes DIRECTION at frame {i}")

# Check smoothstep discontinuity
print("\n" + "="*70)
print("SMOOTHSTEP ANALYSIS")
print("="*70)
print("\nThe issue might be that smoothstep is reset between phases!")
print("In shrink phase: t goes from 0 to 1")
print("In settle phase: t RESETS to 0 and goes to 1 again")
print("This causes smoothstep to reset, creating a discontinuity")

# Calculate what smoothstep does at transition
t_end_shrink = 1.0
smooth_end_shrink = t_end_shrink * t_end_shrink * (3.0 - 2.0 * t_end_shrink)
t_start_settle = 0.0
smooth_start_settle = t_start_settle * t_start_settle * (3.0 - 2.0 * t_start_settle)

print(f"\nAt transition (frame {shrink_frames}):")
print(f"  End of shrink: t=1.0, smooth_t={smooth_end_shrink:.3f}")
print(f"  Start of settle: t=0.0, smooth_t={smooth_start_settle:.3f}")
print(f"  >>> Smoothstep JUMPS from {smooth_end_shrink:.3f} to {smooth_start_settle:.3f}!")

print("\n✅ SOLUTION: Use continuous t across both phases")
print("   Instead of resetting t, continue it from shrink to settle")