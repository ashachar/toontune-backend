#!/usr/bin/env python3
"""Debug why the O isn't properly occluded by the girl"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from utils.segmentation.segment_extractor import extract_foreground_mask
from utils.animations.text_3d_behind_segment_balanced import Text3DBehindSegment

print("="*60)
print("DEBUGGING 'O' OCCLUSION ISSUE")
print("="*60)
print("Finding why the O isn't behind the girl...\n")

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

print(f"Loaded {len(frames)} frames at {W}x{H}")

# Based on the image, the O appears to be around the center-right
# Let's check frames 20-35 where text should be passing behind
print("\nStep 1: Finding the O position in critical frames...")

# Create animation to get text positions
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

# Analyze frames 20-35
critical_frames = range(20, 36)
o_positions = {}

for frame_idx in critical_frames:
    # Generate the frame
    result = anim.generate_frame(frame_idx, frames[frame_idx])
    
    # Try to locate the O (it's roughly in the middle of "HELLO WORLD")
    # The O is approximately at position 5-6 in the 11-character string
    # So it's about 45-55% across the text width
    
    # Save frame for inspection
    if frame_idx in [25, 28, 30, 32]:
        cv2.imwrite(f"debug_o_frame_{frame_idx}.png", cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
        print(f"  Saved frame {frame_idx}")

print("\nStep 2: Analyzing mask at O position (frame 30 looks critical)...")

# Focus on frame 30 based on the image
frame_idx = 30
frame = frames[frame_idx]

# Extract and analyze mask
mask = extract_foreground_mask(frame)
print(f"\nFrame {frame_idx} mask analysis:")
print(f"  Total mask coverage: {100 * np.sum(mask > 128) / mask.size:.1f}%")

# The O appears to be around x=550-650, y=250-350 based on the image
o_region_x = 550
o_region_y = 250
o_region_w = 100
o_region_h = 100

# Extract O region
o_mask_region = mask[o_region_y:o_region_y+o_region_h, 
                     o_region_x:o_region_x+o_region_w]
o_frame_region = frame[o_region_y:o_region_y+o_region_h,
                       o_region_x:o_region_x+o_region_w]

print(f"\nO region ({o_region_x}, {o_region_y}) analysis:")
print(f"  Mask coverage in O area: {100 * np.sum(o_mask_region > 128) / o_mask_region.size:.1f}%")

# Save O region analysis
fig = np.zeros((o_region_h*2, o_region_w*3, 3), dtype=np.uint8)
fig[:o_region_h, :o_region_w] = o_frame_region
fig[:o_region_h, o_region_w:o_region_w*2] = np.stack([o_mask_region]*3, axis=2)
fig[:o_region_h, o_region_w*2:] = cv2.addWeighted(o_frame_region, 0.7, 
                                                   np.stack([o_mask_region]*3, axis=2), 0.3, 0)

cv2.imwrite("debug_o_region.png", cv2.cvtColor(fig, cv2.COLOR_RGB2BGR))
print("  Saved O region analysis to debug_o_region.png")

print("\nStep 3: Testing different mask processing for O region...")

# Test different mask processing approaches
approaches = [
    ("original", lambda m: m),
    ("small_dilate", lambda m: cv2.dilate(m, np.ones((3, 3), np.uint8), iterations=1)),
    ("balanced_5x5", lambda m: cv2.dilate(cv2.GaussianBlur(m, (3, 3), 0), 
                                          np.ones((5, 5), np.uint8), iterations=1)),
    ("stronger_7x7", lambda m: cv2.dilate(cv2.GaussianBlur(m, (5, 5), 0), 
                                          np.ones((7, 7), np.uint8), iterations=1)),
    ("targeted_close", lambda m: cv2.dilate(
        cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)),
        np.ones((5, 5), np.uint8), iterations=1)),
]

for name, process_fn in approaches:
    processed = process_fn(mask.copy())
    processed = (processed > 100).astype(np.uint8) * 255
    
    # Check O region
    o_region_processed = processed[o_region_y:o_region_y+o_region_h,
                                   o_region_x:o_region_x+o_region_w]
    coverage = 100 * np.sum(o_region_processed > 128) / o_region_processed.size
    
    print(f"  {name:15s}: {coverage:5.1f}% coverage in O region")
    
    # Save the best ones
    if name in ["balanced_5x5", "stronger_7x7", "targeted_close"]:
        cv2.imwrite(f"debug_o_mask_{name}.png", processed)

print("\nStep 4: Checking girl's position relative to O...")

# The girl sitting appears to be around x=500-700, y=200-400
girl_region = mask[200:400, 500:700]
girl_coverage = 100 * np.sum(girl_region > 128) / girl_region.size
print(f"  Girl region (500-700, 200-400) coverage: {girl_coverage:.1f}%")

if girl_coverage < 50:
    print("  ⚠️ PROBLEM: Girl is not being properly detected by mask!")
    print("  This is why the O shows through - the mask doesn't cover her body")

print("\nStep 5: Visualizing the problem...")

# Create a comprehensive visualization
full_viz = np.zeros((H*2, W*2, 3), dtype=np.uint8)

# Top-left: Original frame
full_viz[:H, :W] = frame

# Top-right: Current mask
mask_rgb = np.stack([mask]*3, axis=2)
full_viz[:H, W:] = mask_rgb

# Bottom-left: Mask overlay showing problem
overlay = frame.copy()
# Highlight O region in red
cv2.rectangle(overlay, (o_region_x, o_region_y), 
              (o_region_x+o_region_w, o_region_y+o_region_h), (255, 0, 0), 2)
# Highlight girl region in green  
cv2.rectangle(overlay, (500, 200), (700, 400), (0, 255, 0), 2)
full_viz[H:, :W] = overlay

# Bottom-right: Enhanced mask
enhanced_mask = cv2.dilate(
    cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)),
    np.ones((7, 7), np.uint8), iterations=1
)
enhanced_mask = (enhanced_mask > 100).astype(np.uint8) * 255
full_viz[H:, W:] = np.stack([enhanced_mask]*3, axis=2)

# Add labels
cv2.putText(full_viz, "Frame", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
cv2.putText(full_viz, "Current Mask", (W+10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
cv2.putText(full_viz, "O region (red), Girl (green)", (10, H+30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
cv2.putText(full_viz, "Enhanced Mask", (W+10, H+30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

cv2.imwrite("debug_o_full_analysis.png", cv2.cvtColor(full_viz, cv2.COLOR_RGB2BGR))

print("\n" + "="*60)
print("DIAGNOSIS")
print("="*60)

if girl_coverage < 50:
    print("\n❌ MAIN ISSUE: The girl sitting is not being detected by rembg!")
    print("   The mask has poor coverage in the girl's area")
    print("   This causes the O to show through when it should be hidden")
    print("\nSOLUTION: Need more aggressive masking specifically for seated figures")
else:
    print("\n⚠️ Partial detection issue")
    print("   The mask partially covers the girl but not completely")
    print("   Need better coverage to fully occlude the O")

print("\n📸 Debug outputs:")
print("  • debug_o_frame_*.png - Key frames")
print("  • debug_o_region.png - O region closeup")
print("  • debug_o_mask_*.png - Different mask approaches")
print("  • debug_o_full_analysis.png - Complete visualization")
print("\nThe O should be behind the girl but mask doesn't cover her properly!")