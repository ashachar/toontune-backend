#!/usr/bin/env python3
"""Check actual text width and positioning"""

from PIL import Image, ImageDraw, ImageFont

# Create test image
img = Image.new('RGBA', (1280, 720), (0, 0, 0, 0))
draw = ImageDraw.Draw(img)

# Load font
try:
    font = ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc', 55)
except:
    font = ImageFont.load_default()

# Measure "Yes,"
text = "Yes,"
bbox = draw.textbbox((0, 0), text, font=font)
width = bbox[2] - bbox[0]
height = bbox[3] - bbox[1]

print(f"Text: '{text}'")
print(f"Font size: 55")
print(f"Actual width: {width}px")
print(f"Height: {height}px")

# Draw at x=211 to visualize
draw.text((211, 360), text, fill=(255, 255, 255, 255), font=font)

# Draw a red line at x=211 to show start position
for y in range(720):
    img.putpixel((211, y), (255, 0, 0, 255))

# Draw person's approximate face position (from detection)
draw.rectangle((537, 30, 736, 229), outline=(0, 255, 0, 128), width=2)

img.save('./outputs/text_position_check.png')
print("\nVisualization saved to ./outputs/text_position_check.png")
print(f"\nWith text starting at x=211:")
print(f"  Text spans from x=211 to x={211 + width}")
print(f"  Face detected from x=537 to x=736")