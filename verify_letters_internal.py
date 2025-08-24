#!/usr/bin/env python3
"""Verify individual letter positions using internal animation state"""

import cv2
import numpy as np
from utils.animations.text_3d_motion_dissolve_correct import Text3DMotionDissolve

print("="*80)
print("VERIFYING LETTER POSITIONS FROM INTERNAL STATE")
print("="*80)

# Load video for background
video_path = "test_element_3sec.mp4"
cap = cv2.VideoCapture(video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

frames = []
for i in range(50):
    ret, frame = cap.read()
    if not ret:
        break
    frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
cap.release()

# Create animation to inspect internal letter positions
class DebugLetterAnimation(Text3DMotionDissolve):
    """Extended class to expose internal letter positions"""
    
    def get_motion_text_bounds(self, frame_num):
        """Get the bounds of text in motion phase"""
        # Calculate parameters for this frame
        t_global = frame_num / max(self.motion_frames - 1, 1)
        smooth_t_global = self._smoothstep(t_global)
        
        # Calculate scale
        shrink_progress = self.shrink_duration / self.motion_duration
        if smooth_t_global <= shrink_progress:
            local_t = smooth_t_global / shrink_progress
            scale = self.start_scale - local_t * (self.start_scale - self.end_scale)
        else:
            local_t = (smooth_t_global - shrink_progress) / (1.0 - shrink_progress)
            scale = self.end_scale - local_t * (self.end_scale - self.final_scale)
        
        # Get text dimensions at this scale
        font_px = int(self.font_size * scale)
        
        # Estimate text width (approximate)
        text_width = font_px * len(self.text) * 0.8  # Rough estimate
        
        # Calculate position
        cx, cy = self.center_position
        start_y = cy - self.resolution[1] * 0.15
        end_y = cy
        
        # Get anchor from rendering
        _, (anchor_x, anchor_y) = self._render_3d_text(
            self.text, scale, 0.2, False, self.transition_scale
        )
        
        pos_x = cx - anchor_x
        pos_y = start_y + smooth_t_global * (end_y - start_y) - anchor_y
        
        return {
            'scale': scale,
            'center': (cx, cy),
            'top_left': (pos_x, pos_y),
            'anchor': (anchor_x, anchor_y),
            'approx_width': text_width
        }
    
    def get_dissolve_letter_positions(self):
        """Get the pre-calculated letter positions for dissolve phase"""
        return list(zip(self.text, self.letter_positions))

# Create debug animation
print("\nCreating animation with debug access...")
anim = DebugLetterAnimation(
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

print("\n" + "="*80)
print("MOTION PHASE FINAL STATE")
print("="*80)

# Get motion phase final state
transition_frame = anim.motion_frames
last_motion_state = anim.get_motion_text_bounds(transition_frame - 1)

print(f"\nLast motion frame (frame {transition_frame - 1}):")
print(f"  Scale: {last_motion_state['scale']:.4f}")
print(f"  Center target: {last_motion_state['center']}")
print(f"  Top-left position: ({last_motion_state['top_left'][0]:.1f}, {last_motion_state['top_left'][1]:.1f})")
print(f"  Anchor offset: ({last_motion_state['anchor'][0]:.1f}, {last_motion_state['anchor'][1]:.1f})")
print(f"  Approx text width: {last_motion_state['approx_width']:.1f}")

# Calculate where letters SHOULD be in motion phase
expected_text_center_x = last_motion_state['center'][0]
expected_text_left = expected_text_center_x - last_motion_state['approx_width'] / 2

print(f"\n  Expected text bounds:")
print(f"    Left edge: ~{expected_text_left:.1f}")
print(f"    Center: {expected_text_center_x:.1f}")
print(f"    Right edge: ~{expected_text_left + last_motion_state['approx_width']:.1f}")

print("\n" + "="*80)
print("DISSOLVE PHASE LETTER POSITIONS")
print("="*80)

# Get dissolve phase letter positions
letter_positions = anim.get_dissolve_letter_positions()

print(f"\nIndividual letter positions in dissolve phase:")
print("-" * 60)
print("Letter | Position (x, y) | Expected X Range")
print("-" * 60)

x_positions = []
for letter, pos in letter_positions:
    if pos is not None:
        x_positions.append(pos[0])
        # Check if position is reasonable
        status = "✅" if 0 < pos[0] < W else "⚠️"
        print(f"  '{letter}'  | ({pos[0]:4d}, {pos[1]:4d})    | {status}")
    else:
        print(f"  '{letter}'  | (space)          |")

if x_positions:
    min_x = min(x_positions)
    max_x = max(x_positions)
    # Estimate letter width
    avg_letter_width = (max_x - min_x) / len([c for c in anim.text if c != ' '])
    max_x_with_width = max_x + avg_letter_width  # Add last letter width
    
    center_x = (min_x + max_x_with_width) / 2
    
    print(f"\nActual dissolve letter bounds:")
    print(f"  First letter X: {min_x}")
    print(f"  Last letter X: {max_x}")
    print(f"  Estimated right edge: {max_x_with_width:.1f}")
    print(f"  Estimated center: {center_x:.1f}")
    
    # Compare with expected
    print(f"\nComparison with motion phase:")
    print(f"  Motion text center: {expected_text_center_x:.1f}")
    print(f"  Dissolve text center: {center_x:.1f}")
    print(f"  Center shift: {center_x - expected_text_center_x:.1f} pixels")
    
    if abs(center_x - expected_text_center_x) > 50:
        print("\n❌ SIGNIFICANT MISMATCH!")
        print("   Letters are not centered at the same position as motion phase")
    elif abs(center_x - expected_text_center_x) > 10:
        print("\n⚠️ MODERATE MISMATCH")
        print("   Letters are slightly off-center")
    else:
        print("\n✅ GOOD ALIGNMENT")
        print("   Letters are properly centered")

# Generate actual frames to verify visually
print("\n" + "="*80)
print("GENERATING FRAMES FOR VISUAL VERIFICATION")
print("="*80)

last_motion = anim.generate_frame(transition_frame - 1, frames[transition_frame - 1])
first_dissolve = anim.generate_frame(transition_frame, frames[transition_frame])

if last_motion.shape[2] == 4:
    last_motion = last_motion[:, :, :3]
if first_dissolve.shape[2] == 4:
    first_dissolve = first_dissolve[:, :, :3]

# Create comparison image with grid
comparison = np.hstack([last_motion, first_dissolve])

# Add center lines
from PIL import Image, ImageDraw
img = Image.fromarray(comparison)
draw = ImageDraw.Draw(img)

# Draw center lines
center_x = W // 2
draw.line([(center_x, 0), (center_x, H)], fill=(0, 255, 0), width=2)
draw.line([(W + center_x, 0), (W + center_x, H)], fill=(0, 255, 0), width=2)

# Draw expected letter positions on dissolve side
if x_positions:
    # Mark first and last letter positions
    draw.line([(W + min_x, 0), (W + min_x, H)], fill=(255, 0, 0), width=1)
    draw.line([(W + max_x_with_width, 0), (W + max_x_with_width, H)], fill=(255, 0, 0), width=1)
    
    # Mark center
    draw.line([(W + int(center_x), 0), (W + int(center_x), H)], fill=(255, 255, 0), width=2)

comparison = np.array(img)
cv2.imwrite('letter_position_analysis.png', cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))
print("\nSaved: letter_position_analysis.png")
print("  Green lines: Frame centers (should align with text)")
print("  Red lines: Letter bounds in dissolve")
print("  Yellow line: Calculated center of dissolve letters")

print("\n✅ INTERNAL STATE VERIFICATION COMPLETE")