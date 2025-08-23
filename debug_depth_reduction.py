#!/usr/bin/env python3
"""Debug why depth suddenly reduces after passing foreground"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from utils.animations.text_3d_behind_segment_final import Text3DBehindSegment

print("="*60)
print("DEBUGGING SUDDEN DEPTH REDUCTION")
print("="*60)

video_path = "test_element_3sec.mp4"
cap = cv2.VideoCapture(video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Load frames
frames = []
for i in range(45):
    ret, frame = cap.read()
    if not ret:
        break
    frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
cap.release()

# The issue appears to be around frames 25-30
# Let's examine the animation parameters at these frames
print("\nAnalyzing animation parameters around the transition...")

# Create animation instance
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
    shrink_duration=0.6,
    wiggle_duration=0.15,
    center_position=(W//2, H//2),
    shadow_offset=6,
    outline_width=2,
    perspective_angle=0,
    supersample_factor=2,
    debug=False,
)

# Check the transition point
shrink_frames = int(0.6 * fps)  # 36 frames
print(f"Shrink phase: frames 0-{shrink_frames-1}")
print(f"Wiggle phase: frames {shrink_frames}-44")

# The issue is likely around the 40-60% mark of shrink phase
# which is frames 14-22 (40% of 36 = 14.4, 60% = 21.6)
critical_frames = range(18, 28)

print("\nFrame-by-frame analysis:")
print("Frame | Phase  | t     | Scale | Alpha | Behind | Depth")
print("-" * 60)

for frame_idx in critical_frames:
    # Calculate phase and parameters
    if frame_idx < shrink_frames:
        phase = "shrink"
        t = frame_idx / max(shrink_frames - 1, 1)
        scale = 2.0 - t * (2.0 - 1.0)
        
        if t < 0.4:
            alpha = 1.0
            is_behind = False
        elif t < 0.6:
            fade_t = (t - 0.4) / 0.2
            alpha = 1.0 - fade_t * 0.4
            is_behind = False
        else:
            fade_t = (t - 0.6) / 0.4
            k = 3.0
            alpha = 0.6 - 0.4 * (1 - np.exp(-k * fade_t)) / (1 - np.exp(-k))
            is_behind = True
    else:
        phase = "wiggle"
        t = (frame_idx - shrink_frames) / max(9, 1)
        wiggle_amount = np.sin(t * np.pi * 4) * 0.02
        scale = 1.0 * (1.0 + wiggle_amount)
        alpha = 0.2
        is_behind = True
    
    # Calculate depth offset
    depth_offset = 3  # Base depth offset
    depth_off_scaled = depth_offset * scale * 2  # At supersample
    reduced_depth = depth_off_scaled * 0.2  # 80% reduction
    
    print(f"{frame_idx:5d} | {phase:6s} | {t:.3f} | {scale:.3f} | {alpha:.3f} | {is_behind:5s} | {reduced_depth:.1f}")
    
    # Generate and save critical frames
    if frame_idx in [20, 22, 24, 26]:
        result = anim.generate_frame(frame_idx, frames[frame_idx])
        cv2.imwrite(f"depth_debug_frame_{frame_idx}.png", cv2.cvtColor(result, cv2.COLOR_RGB2BGR))

print("\n" + "="*60)
print("ISSUE IDENTIFIED!")
print("="*60)
print("\nThe depth is tied to SCALE, which continues shrinking!")
print("As scale goes from 2.0 → 1.0, depth also reduces")
print("This causes the visible depth reduction after passing foreground")
print("\nSOLUTION: Maintain constant depth offset regardless of scale")
print("when the text is behind the foreground!")

# Test with fixed depth
print("\nTesting with constant depth when behind...")

class FixedDepthTest:
    def calculate_depth(self, frame_idx, base_depth=3):
        shrink_frames = 36
        
        if frame_idx < shrink_frames:
            t = frame_idx / max(shrink_frames - 1, 1)
            scale = 2.0 - t * (2.0 - 1.0)
            
            # Check if behind
            if t >= 0.6:  # Behind
                # Use constant depth when behind
                return base_depth * 2.0 * 0.2  # Use start_scale for consistent depth
            else:
                # Normal scaling when in front
                return base_depth * scale * 0.2
        else:
            # Wiggle phase - keep constant depth
            return base_depth * 2.0 * 0.2

test = FixedDepthTest()
print("\nFixed depth calculation:")
print("Frame | Depth (current) | Depth (fixed)")
print("-" * 40)
for frame_idx in [18, 20, 22, 24, 26, 28]:
    if frame_idx < 36:
        t = frame_idx / 35
        scale = 2.0 - t * (2.0 - 1.0)
        current_depth = 3 * scale * 0.2
    else:
        current_depth = 3 * 1.0 * 0.2
    
    fixed_depth = test.calculate_depth(frame_idx)
    print(f"{frame_idx:5d} | {current_depth:14.2f} | {fixed_depth:13.2f}")

print("\nThe fix: When is_behind=True, use constant depth offset")
print("instead of scaling it down with the shrinking animation!")