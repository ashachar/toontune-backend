#!/usr/bin/env python3
"""Debug the masking issue - understand what's happening"""

import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from utils.animations.text_3d_behind_segment import Text3DBehindSegment
from utils.segmentation.segment_extractor import extract_foreground_mask

print("="*60)
print("DEBUGGING MASKING ISSUE")
print("="*60)

# Load a single frame to debug with
video_path = "test_element_3sec.mp4"
cap = cv2.VideoCapture(video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Get frame 48 - where the W occlusion happens
cap.set(cv2.CAP_PROP_POS_FRAMES, 48)
ret, frame = cap.read()
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
cap.release()

print(f"\nFrame dimensions: {W}x{H}")
print("Testing frame 48 (0.8 seconds) - W occlusion moment\n")

# Extract the mask from this frame
print("1. Extracting foreground mask...")
mask = extract_foreground_mask(frame_rgb)
print(f"   Mask shape: {mask.shape}")
print(f"   Mask unique values: {np.unique(mask)}")
print(f"   Foreground pixels: {np.sum(mask > 128):,} / {mask.size:,}")

# Save mask for inspection
cv2.imwrite("debug_1_extracted_mask.png", mask)

# Test the original animation class
print("\n2. Testing ORIGINAL animation class...")
anim = Text3DBehindSegment(
    duration=1.0,
    fps=fps,
    resolution=(W, H),
    text="HELLO WORLD",
    segment_mask=None,  # Dynamic
    font_size=140,
    text_color=(255, 220, 0),
    depth_color=(200, 170, 0),
    depth_layers=10,
    depth_offset=3,
    start_scale=1.8,
    end_scale=0.9,
    phase1_duration=0.4,
    phase2_duration=0.5,
    phase3_duration=0.1,
    center_position=(W//2, H//2),
    shadow_offset=6,
    outline_width=2,
    perspective_angle=0,
    supersample_factor=2,
    debug=True,  # Enable debug output!
)

# Generate frame 48
print("\n3. Generating frame with debug output...")
result_frame = anim.generate_frame(48, frame_rgb)

# Convert and save
if result_frame.shape[2] == 4:
    result_frame = result_frame[:, :, :3]
cv2.imwrite("debug_2_result_frame.png", cv2.cvtColor(result_frame, cv2.COLOR_RGB2BGR))

print("\n4. Analyzing text placement...")
# Try to understand where the text is being placed

# Create a simple test without any masking
print("\n5. Testing WITHOUT masking (is_behind=False)...")
class SimpleText3D:
    def __init__(self):
        self.font_size = 140
        self.text_color = (255, 220, 0)
        self.depth_color = (200, 170, 0)
        self.resolution = (W, H)
        
    def render_simple(self, text, scale):
        from PIL import ImageDraw, ImageFont
        
        # Simple render without masking
        font_size = int(self.font_size * scale)
        
        # Create image
        img = Image.new("RGBA", self.resolution, (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        
        # Try to load font
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", font_size)
        except:
            font = ImageFont.load_default()
        
        # Draw text centered
        bbox = draw.textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        
        x = (self.resolution[0] - text_w) // 2
        y = (self.resolution[1] - text_h) // 2
        
        # Draw depth layers
        for i in range(5, 0, -1):
            offset = i * 2
            color = tuple(int(c * 0.8) for c in self.text_color)
            draw.text((x + offset, y - offset), text, font=font, fill=(*color, 255))
        
        # Draw main text
        draw.text((x, y), text, font=font, fill=(*self.text_color, 255))
        
        return np.array(img)

simple = SimpleText3D()
simple_result = simple.render_simple("HELLO WORLD", 1.2)

# Composite on frame
composite = frame_rgb.copy()
if simple_result.shape[2] == 4:
    alpha = simple_result[:, :, 3:4] / 255.0
    composite = composite * (1 - alpha) + simple_result[:, :, :3] * alpha
    composite = composite.astype(np.uint8)

cv2.imwrite("debug_3_simple_no_mask.png", cv2.cvtColor(composite, cv2.COLOR_RGB2BGR))

print("\n6. Testing mask application on simple text...")
# Apply mask to simple text
masked_simple = simple_result.copy()
if masked_simple.shape[2] == 4:
    # Apply mask to alpha channel
    alpha = masked_simple[:, :, 3].astype(np.float32)
    alpha *= (1.0 - (mask.astype(np.float32) / 255.0))
    masked_simple[:, :, 3] = alpha.astype(np.uint8)

# Composite
composite2 = frame_rgb.copy()
if masked_simple.shape[2] == 4:
    alpha = masked_simple[:, :, 3:4] / 255.0
    composite2 = composite2 * (1 - alpha) + masked_simple[:, :, :3] * alpha
    composite2 = composite2.astype(np.uint8)

cv2.imwrite("debug_4_simple_with_mask.png", cv2.cvtColor(composite2, cv2.COLOR_RGB2BGR))

print("\n" + "="*60)
print("DEBUG OUTPUT SAVED:")
print("="*60)
print("1. debug_1_extracted_mask.png - The foreground mask")
print("2. debug_2_result_frame.png - Result from original class")
print("3. debug_3_simple_no_mask.png - Simple 3D text without mask")
print("4. debug_4_simple_with_mask.png - Simple 3D text with mask")
print("\nCheck these images to understand the issue!")