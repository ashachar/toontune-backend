#!/usr/bin/env python3
"""Debug why W is still visible when it should be behind the head"""

import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from utils.animations.text_3d_behind_segment_backwards_only import Text3DBehindSegment
from utils.segmentation.segment_extractor import extract_foreground_mask

print("="*60)
print("DEBUGGING W VISIBILITY ISSUE")
print("="*60)
print("Finding the exact frame where W should be fully hidden...\n")

video_path = "test_element_3sec.mp4"
cap = cv2.VideoCapture(video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Based on the image, this looks like it's around frame 20-30 in our animation
# Let's check multiple frames to find the exact one
test_frames = range(15, 35)

# Load frames
frames = []
for i in range(45):
    ret, frame = cap.read()
    if not ret:
        break
    frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
cap.release()

# Create animation with debug enabled
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
    debug=True,  # Enable debug
)

print("Analyzing frames to find where W should be behind head...")
for frame_idx in test_frames:
    if frame_idx >= len(frames):
        break
    
    # Generate frame
    result = anim.generate_frame(frame_idx, frames[frame_idx])
    
    # Save this frame for inspection
    if frame_idx in [20, 25, 30]:
        cv2.imwrite(f"debug_frame_{frame_idx}.png", cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
        
        # Also extract and save the mask for this frame
        mask = extract_foreground_mask(frames[frame_idx])
        cv2.imwrite(f"debug_mask_{frame_idx}.png", mask)
        
        print(f"\nFrame {frame_idx}: Saved frame and mask")

print("\n" + "="*60)
print("Now let's analyze frame 25 in detail (likely the problematic one)")
print("="*60)

# Focus on frame 25
frame_idx = 25
frame = frames[frame_idx]

# Extract mask
print("\n1. Extracting foreground mask...")
mask = extract_foreground_mask(frame)
print(f"   Mask shape: {mask.shape}")
print(f"   Unique values: {np.unique(mask)}")

# Check the W region specifically
# The W appears to be around x=640, y=250 based on the image
w_region_x = 640
w_region_y = 250
region_size = 150

# Extract region around W
w_region = frame[w_region_y:w_region_y+region_size, 
                 w_region_x:w_region_x+region_size]
mask_region = mask[w_region_y:w_region_y+region_size,
                   w_region_x:w_region_x+region_size]

print(f"\n2. W region analysis:")
print(f"   Region center: ({w_region_x}, {w_region_y})")
print(f"   Mask values in region: min={mask_region.min()}, max={mask_region.max()}")
print(f"   Foreground pixels in region: {np.sum(mask_region > 128)} / {mask_region.size}")

# Save zoomed regions
cv2.imwrite("debug_w_region.png", cv2.cvtColor(w_region, cv2.COLOR_RGB2BGR))
cv2.imwrite("debug_w_mask.png", mask_region)

# Create overlay to show mask on image
overlay = frame.copy()
mask_colored = np.zeros_like(frame)
mask_colored[:, :, 1] = mask  # Green channel for mask
overlay = cv2.addWeighted(overlay, 0.7, mask_colored, 0.3, 0)
cv2.imwrite("debug_overlay.png", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

print("\n3. Testing different mask strengths...")

# Try applying mask with different strengths
from PIL import ImageDraw, ImageFont

# Create test text at the W position
test_img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
draw = ImageDraw.Draw(test_img)
try:
    font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 140)
except:
    font = ImageFont.load_default()

# Draw just W at the position
draw.text((w_region_x, w_region_y), "W", font=font, fill=(255, 220, 0, 255))
test_array = np.array(test_img)

# Test different mask applications
strengths = [0.95, 0.98, 1.0]  # Current, stronger, full
for strength in strengths:
    test_copy = test_array.copy()
    alpha = test_copy[:, :, 3].astype(np.float32)
    alpha *= (1.0 - (mask.astype(np.float32) / 255.0) * strength)
    test_copy[:, :, 3] = alpha.astype(np.uint8)
    
    # Composite
    result = frame.copy()
    if test_copy.shape[2] == 4:
        alpha_3d = test_copy[:, :, 3:4] / 255.0
        result = result * (1 - alpha_3d) + test_copy[:, :, :3] * alpha_3d
        result = result.astype(np.uint8)
    
    cv2.imwrite(f"debug_strength_{int(strength*100)}.png", cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
    print(f"   Saved with {int(strength*100)}% mask strength")

print("\n4. Checking mask dilation...")

# Test different mask dilations
kernel_sizes = [3, 5, 7]
for ksize in kernel_sizes:
    kernel = np.ones((ksize, ksize), np.uint8)
    mask_dilated = cv2.dilate(mask, kernel, iterations=1)
    
    test_copy = test_array.copy()
    alpha = test_copy[:, :, 3].astype(np.float32)
    alpha *= (1.0 - (mask_dilated.astype(np.float32) / 255.0))
    test_copy[:, :, 3] = alpha.astype(np.uint8)
    
    # Composite
    result = frame.copy()
    if test_copy.shape[2] == 4:
        alpha_3d = test_copy[:, :, 3:4] / 255.0
        result = result * (1 - alpha_3d) + test_copy[:, :, :3] * alpha_3d
        result = result.astype(np.uint8)
    
    cv2.imwrite(f"debug_dilate_{ksize}.png", cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
    print(f"   Saved with {ksize}x{ksize} dilation")

print("\n" + "="*60)
print("DEBUG OUTPUTS SAVED:")
print("="*60)
print("\nFrames and masks:")
print("  • debug_frame_20.png, debug_mask_20.png")
print("  • debug_frame_25.png, debug_mask_25.png") 
print("  • debug_frame_30.png, debug_mask_30.png")
print("\nW region analysis:")
print("  • debug_w_region.png - Zoomed W area")
print("  • debug_w_mask.png - Mask in W area")
print("  • debug_overlay.png - Mask overlay on frame")
print("\nMask strength tests:")
print("  • debug_strength_95.png - Current 95% strength")
print("  • debug_strength_98.png - Stronger 98%")
print("  • debug_strength_100.png - Full 100% occlusion")
print("\nMask dilation tests:")
print("  • debug_dilate_3.png - 3x3 dilation")
print("  • debug_dilate_5.png - 5x5 dilation")
print("  • debug_dilate_7.png - 7x7 dilation")
print("\nCheck these to see which approach hides the W properly!")