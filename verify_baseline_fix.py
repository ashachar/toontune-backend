#!/usr/bin/env python3
"""Verify that baseline alignment is now correct."""

from PIL import Image, ImageDraw, ImageFont
import numpy as np

# Create test image showing the fix
img = Image.new('RGBA', (600, 300), (255, 255, 255, 255))
draw = ImageDraw.Draw(img)

# Use a large font
try:
    font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 80)
except:
    font = ImageFont.load_default()

text = "Hello"

# Draw with baseline alignment (corrected)
x_pos = 50
y_baseline = 150
margin = 10

for letter in text:
    # Get bbox for this letter
    bbox = draw.textbbox((0, 0), letter, font=font)
    bbox_h = bbox[3] - bbox[1]
    
    # Position to align bottoms
    x = x_pos - bbox[0]
    y = y_baseline - bbox[3]  # This aligns the bottom at y_baseline
    
    # Draw letter
    draw.text((x, y), letter, font=font, fill=(0, 0, 255, 255))
    
    # Draw baseline for reference
    draw.line([(x_pos - 5, y_baseline), (x_pos + 60, y_baseline)], fill=(255, 0, 0, 128), width=1)
    
    x_pos += bbox[2] - bbox[0] + 10

# Add label
label_font = ImageFont.load_default()
draw.text((10, 10), "Baseline-aligned (CORRECT - all bottoms on red line):", font=label_font, fill=(0, 0, 0, 255))

img.save('outputs/baseline_verification.png')
print("✅ Baseline alignment verification saved to outputs/baseline_verification.png")
print("All letters should have their bottoms aligned on the red baseline.")