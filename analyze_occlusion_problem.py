#!/usr/bin/env python3
"""Analyze the actual occlusion problem at W boundary"""

import cv2
import numpy as np
from PIL import Image, ImageDraw
from utils.animations.text_3d_behind_segment import Text3DBehindSegment
from utils.segmentation.segment_extractor import extract_foreground_mask

print("ANALYZING W OCCLUSION PROBLEM")
print("=" * 50)

# Load critical frame
video_path = "test_element_3sec.mp4"
cap = cv2.VideoCapture(video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

cap.set(cv2.CAP_PROP_POS_FRAMES, 48)
ret, frame = cap.read()
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
cap.release()

# Extract mask
print("1. Extracting mask...")
mask = extract_foreground_mask(frame_rgb)

# Visualize mask boundary near W location
w_region_x = int(W * 0.55)  # Approximate W position
w_region_y = int(H * 0.45)
region_size = 200

# Extract regions
frame_region = frame_rgb[w_region_y:w_region_y+region_size, 
                         w_region_x:w_region_x+region_size]
mask_region = mask[w_region_y:w_region_y+region_size, 
                  w_region_x:w_region_x+region_size]

# Upscale for visibility
scale = 3
frame_region_large = cv2.resize(frame_region, 
                                (region_size*scale, region_size*scale), 
                                interpolation=cv2.INTER_NEAREST)
mask_region_large = cv2.resize(mask_region, 
                               (region_size*scale, region_size*scale), 
                               interpolation=cv2.INTER_NEAREST)

# Create mask overlay
mask_colored = np.zeros((region_size*scale, region_size*scale, 3), dtype=np.uint8)
mask_colored[:, :, 1] = mask_region_large  # Green for foreground
overlay = cv2.addWeighted(frame_region_large, 0.7, mask_colored, 0.3, 0)

cv2.imwrite("occlusion_1_region.png", cv2.cvtColor(frame_region_large, cv2.COLOR_RGB2BGR))
cv2.imwrite("occlusion_2_mask.png", mask_region_large)
cv2.imwrite("occlusion_3_overlay.png", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

print("\n2. Testing different mask edge handling...")

# Create test text at this position
test_img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
draw = ImageDraw.Draw(test_img)

# Simple W at the critical position
from PIL import ImageFont
try:
    font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 140)
except:
    font = ImageFont.load_default()

# Draw just a W with depth layers
text = "W"
x_pos = w_region_x + 50
y_pos = w_region_y + 50

# Depth layers
for i in range(10, 0, -1):
    offset = i * 2
    color = (200 - i*10, 170 - i*10, 0)
    draw.text((x_pos + offset, y_pos - offset), text, font=font, fill=(*color, 255))

# Main W
draw.text((x_pos, y_pos), text, font=font, fill=(255, 220, 0, 255))

test_array = np.array(test_img)

# Apply mask with different edge handling
print("\n3. Testing different mask applications...")

# Method 1: Hard mask (original)
test1 = test_array.copy()
alpha1 = test1[:, :, 3].astype(np.float32)
alpha1 *= (1.0 - (mask.astype(np.float32) / 255.0))
test1[:, :, 3] = alpha1.astype(np.uint8)

# Method 2: Soft mask with blur
mask_soft = cv2.GaussianBlur(mask, (5, 5), 0)
test2 = test_array.copy()
alpha2 = test2[:, :, 3].astype(np.float32)
alpha2 *= (1.0 - (mask_soft.astype(np.float32) / 255.0))
test2[:, :, 3] = alpha2.astype(np.uint8)

# Method 3: Dilated mask (slightly expand foreground)
kernel = np.ones((3, 3), np.uint8)
mask_dilated = cv2.dilate(mask, kernel, iterations=1)
test3 = test_array.copy()
alpha3 = test3[:, :, 3].astype(np.float32)
alpha3 *= (1.0 - (mask_dilated.astype(np.float32) / 255.0))
test3[:, :, 3] = alpha3.astype(np.uint8)

# Composite all tests
def composite(text_array, bg):
    result = bg.copy()
    if text_array.shape[2] == 4:
        alpha = text_array[:, :, 3:4] / 255.0
        result = result * (1 - alpha) + text_array[:, :, :3] * alpha
    return result.astype(np.uint8)

result1 = composite(test1, frame_rgb)
result2 = composite(test2, frame_rgb)
result3 = composite(test3, frame_rgb)

cv2.imwrite("occlusion_4_hard_mask.png", cv2.cvtColor(result1, cv2.COLOR_RGB2BGR))
cv2.imwrite("occlusion_5_soft_mask.png", cv2.cvtColor(result2, cv2.COLOR_RGB2BGR))
cv2.imwrite("occlusion_6_dilated_mask.png", cv2.cvtColor(result3, cv2.COLOR_RGB2BGR))

# Create comparison strip
comparison = np.zeros((H, W*3, 3), dtype=np.uint8)
comparison[:, :W] = result1
comparison[:, W:W*2] = result2
comparison[:, W*2:] = result3

# Add labels
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(comparison, "Hard Mask", (20, 40), font, 1.0, (255, 255, 255), 2)
cv2.putText(comparison, "Soft Mask (Blurred)", (W+20, 40), font, 1.0, (255, 255, 255), 2)
cv2.putText(comparison, "Dilated Mask", (W*2+20, 40), font, 1.0, (255, 255, 255), 2)

cv2.imwrite("occlusion_7_comparison.png", cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))

print("\n" + "="*50)
print("ANALYSIS COMPLETE!")
print("="*50)
print("\nGenerated files:")
print("1. occlusion_1_region.png - Zoomed W region")
print("2. occlusion_2_mask.png - Mask in that region")
print("3. occlusion_3_overlay.png - Mask overlay on frame")
print("4. occlusion_4_hard_mask.png - Original hard mask")
print("5. occlusion_5_soft_mask.png - Soft blurred mask")
print("6. occlusion_6_dilated_mask.png - Slightly expanded mask")
print("7. occlusion_7_comparison.png - Side by side comparison")
print("\nThe issue might be that depth layers extend beyond")
print("the main text and need individual masking!")