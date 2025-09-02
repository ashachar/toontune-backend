#!/usr/bin/env python3
"""Test rendering a phrase directly to see if it's visible"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Create a test frame
frame = cv2.imread('/tmp/orig_5s.png')
h, w = frame.shape[:2]

# Render text directly
text = "TEST CAPTION VISIBLE?"
font_size = 60
try:
    font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", font_size)
except:
    font = ImageFont.load_default()

# Create text image
img = Image.new('RGBA', (w, h), (0, 0, 0, 0))
draw = ImageDraw.Draw(img)

# Draw text in the bottom center
x = w // 2
y = h - 100
draw.text((x, y), text, font=font, fill=(255, 255, 255, 255), anchor="mm")

# Convert to numpy
text_img = np.array(img)

# Composite onto frame
alpha = text_img[:, :, 3:4] / 255.0
composite = frame * (1 - alpha) + text_img[:, :, :3] * alpha
composite = composite.astype(np.uint8)

# Save result
cv2.imwrite('/tmp/test_direct_render.png', composite)

# Compare with original
diff = cv2.absdiff(composite, frame)
changed = np.mean(diff > 10) * 100

print(f"Test caption rendered directly")
print(f"Pixels changed: {changed:.2f}%")
print(f"Output: /tmp/test_direct_render.png")

if changed > 1:
    print("✅ Direct rendering works - text is visible")
else:
    print("❌ Even direct rendering doesn't show text")
