#!/usr/bin/env python3
"""Test to show the baseline alignment issue."""

from PIL import Image, ImageDraw, ImageFont

# Create test image
img = Image.new('RGBA', (800, 200), (255, 255, 255, 255))
draw = ImageDraw.Draw(img)

# Use a large font
try:
    font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 100)
except:
    font = ImageFont.load_default()

text = "Hello"
x = 50

# Draw each letter individually with top-alignment (current behavior)
for i, letter in enumerate(text):
    bbox = draw.textbbox((0, 0), letter, font=font)
    # Current behavior: align tops
    y = 50 - bbox[1]  # This makes all tops at y=50
    draw.text((x, y), letter, font=font, fill=(255, 0, 0, 255))
    x += bbox[2] - bbox[0] + 10

# Draw the same text with baseline alignment (correct behavior)
x = 50
y_baseline = 150

# First, find the maximum ascent (from baseline to top)
max_ascent = 0
for letter in text:
    bbox = draw.textbbox((0, y_baseline), letter, font=font)
    ascent = y_baseline - bbox[1]
    max_ascent = max(max_ascent, ascent)

# Draw with baseline alignment
for i, letter in enumerate(text):
    bbox = draw.textbbox((0, 0), letter, font=font)
    # Correct behavior: use baseline
    draw.text((x, y_baseline), letter, font=font, fill=(0, 0, 255, 255))
    x += bbox[2] - bbox[0] + 10

# Add labels
label_font = ImageFont.load_default()
draw.text((10, 10), "Top-aligned (WRONG - current behavior):", font=label_font, fill=(0, 0, 0, 255))
draw.text((10, 120), "Baseline-aligned (CORRECT):", font=label_font, fill=(0, 0, 0, 255))

img.save('outputs/baseline_test.png')
print("Saved comparison to outputs/baseline_test.png")
print("Red text: top-aligned (current wrong behavior)")
print("Blue text: baseline-aligned (correct behavior)")