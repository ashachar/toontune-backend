#!/usr/bin/env python3
"""Debug mask coordinate mapping issue"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from utils.segmentation.segment_extractor import extract_foreground_mask

print("DEBUGGING MASK COORDINATE MAPPING")
print("-" * 50)

# Load frame
video_path = "test_element_3sec.mp4"
cap = cv2.VideoCapture(video_path)
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

cap.set(cv2.CAP_PROP_POS_FRAMES, 48)
ret, frame = cap.read()
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
cap.release()

print(f"Frame size: {W}x{H}")

# Extract mask
mask = extract_foreground_mask(frame_rgb)
print(f"Mask shape: {mask.shape}")

# Create a simple text at different resolutions
text = "HELLO WORLD"
font_size = 140
supersample = 2

# 1. Original resolution text
print("\n1. Creating text at original resolution...")
img_orig = Image.new("RGBA", (W, H), (0, 0, 0, 0))
draw_orig = ImageDraw.Draw(img_orig)
try:
    font_orig = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", font_size)
except:
    font_orig = ImageFont.load_default()

# Center text
bbox = draw_orig.textbbox((0, 0), text, font=font_orig)
text_w = bbox[2] - bbox[0]
text_h = bbox[3] - bbox[1]
x_orig = (W - text_w) // 2
y_orig = (H - text_h) // 2

# Draw with depth
for i in range(5, 0, -1):
    offset = i * 3
    draw_orig.text((x_orig + offset, y_orig - offset), text, 
                   font=font_orig, fill=(200, 170, 0, 255))
draw_orig.text((x_orig, y_orig), text, font=font_orig, fill=(255, 220, 0, 255))

orig_array = np.array(img_orig)
print(f"Original text position: ({x_orig}, {y_orig}), size: {text_w}x{text_h}")

# 2. Supersampled text
print("\n2. Creating text at supersampled resolution...")
ss_W = W * supersample
ss_H = H * supersample
img_ss = Image.new("RGBA", (ss_W, ss_H), (0, 0, 0, 0))
draw_ss = ImageDraw.Draw(img_ss)
try:
    font_ss = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", font_size * supersample)
except:
    font_ss = ImageFont.load_default()

# Measure at supersampled size
bbox_ss = draw_ss.textbbox((0, 0), text, font=font_ss)
text_w_ss = bbox_ss[2] - bbox_ss[0]
text_h_ss = bbox_ss[3] - bbox_ss[1]
x_ss = (ss_W - text_w_ss) // 2
y_ss = (ss_H - text_h_ss) // 2

# Draw with depth at supersample
for i in range(5, 0, -1):
    offset = i * 3 * supersample
    draw_ss.text((x_ss + offset, y_ss - offset), text, 
                 font=font_ss, fill=(200, 170, 0, 255))
draw_ss.text((x_ss, y_ss), text, font=font_ss, fill=(255, 220, 0, 255))

# Downsample
img_ss_down = img_ss.resize((W, H), Image.Resampling.LANCZOS)
ss_array = np.array(img_ss_down)
print(f"Supersampled text position: ({x_ss}, {y_ss}), size: {text_w_ss}x{text_h_ss}")
print(f"After downsample: ({x_ss//supersample}, {y_ss//supersample})")

# 3. Apply mask to original resolution text
print("\n3. Applying mask to original resolution text...")
masked_orig = orig_array.copy()
alpha = masked_orig[:, :, 3].astype(np.float32)
alpha *= (1.0 - (mask.astype(np.float32) / 255.0))
masked_orig[:, :, 3] = alpha.astype(np.uint8)

# 4. Apply mask to downsampled supersampled text
print("4. Applying mask to downsampled supersampled text...")
masked_ss = ss_array.copy()
alpha_ss = masked_ss[:, :, 3].astype(np.float32)
alpha_ss *= (1.0 - (mask.astype(np.float32) / 255.0))
masked_ss[:, :, 3] = alpha_ss.astype(np.uint8)

# 5. Try to apply mask BEFORE downsampling (incorrect approach from my fix)
print("\n5. Testing mask upsampling (the problematic approach)...")
# Upsample mask to supersampled resolution
mask_ss = cv2.resize(mask, (ss_W, ss_H), interpolation=cv2.INTER_LINEAR)
print(f"Upsampled mask shape: {mask_ss.shape}")

# Apply to supersampled image
img_ss_array = np.array(img_ss)
alpha_ss_pre = img_ss_array[:, :, 3].astype(np.float32)
alpha_ss_pre *= (1.0 - (mask_ss.astype(np.float32) / 255.0))
img_ss_array[:, :, 3] = alpha_ss_pre.astype(np.uint8)
img_ss_masked = Image.fromarray(img_ss_array)
img_ss_masked_down = img_ss_masked.resize((W, H), Image.Resampling.LANCZOS)
ss_masked_array = np.array(img_ss_masked_down)

# Save all results
def composite_on_frame(text_array, bg):
    result = bg.copy()
    if text_array.shape[2] == 4:
        alpha = text_array[:, :, 3:4] / 255.0
        result = result * (1 - alpha) + text_array[:, :, :3] * alpha
        result = result.astype(np.uint8)
    return result

# Save comparisons
cv2.imwrite("mask_debug_1_original.png", cv2.cvtColor(composite_on_frame(orig_array, frame_rgb), cv2.COLOR_RGB2BGR))
cv2.imwrite("mask_debug_2_supersampled.png", cv2.cvtColor(composite_on_frame(ss_array, frame_rgb), cv2.COLOR_RGB2BGR))
cv2.imwrite("mask_debug_3_orig_masked.png", cv2.cvtColor(composite_on_frame(masked_orig, frame_rgb), cv2.COLOR_RGB2BGR))
cv2.imwrite("mask_debug_4_ss_masked_after.png", cv2.cvtColor(composite_on_frame(masked_ss, frame_rgb), cv2.COLOR_RGB2BGR))
cv2.imwrite("mask_debug_5_ss_masked_before.png", cv2.cvtColor(composite_on_frame(ss_masked_array, frame_rgb), cv2.COLOR_RGB2BGR))
cv2.imwrite("mask_debug_6_mask.png", mask)
cv2.imwrite("mask_debug_7_mask_upsampled.png", mask_ss)

print("\n" + "="*50)
print("OUTPUTS:")
print("1. mask_debug_1_original.png - Original resolution")
print("2. mask_debug_2_supersampled.png - Supersampled")
print("3. mask_debug_3_orig_masked.png - Original + mask")
print("4. mask_debug_4_ss_masked_after.png - SS + mask after downsample")
print("5. mask_debug_5_ss_masked_before.png - SS + mask before downsample")
print("6. mask_debug_6_mask.png - Original mask")
print("7. mask_debug_7_mask_upsampled.png - Upsampled mask")
print("\nThe issue is likely in #5 - applying mask at wrong resolution!")