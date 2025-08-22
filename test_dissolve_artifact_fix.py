#!/usr/bin/env python3
"""Quick test to verify the letter frame dissolve artifact fix."""

import os
os.environ['FRAME_DISSOLVE_DEBUG'] = '1'

import cv2
import numpy as np
from utils.animations.word_dissolve import WordDissolve

# Create a simple test video
def create_test_video(filename, width=640, height=480, fps=30, duration=1):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, fps, (width, height))
    
    frames = int(fps * duration)
    for i in range(frames):
        # Create a gradient background
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:, :] = [100, 100, 100]  # Gray background
        out.write(frame)
    
    out.release()
    print(f"Created test video: {filename}")
    return filename

# Create test video
test_video = "test_artifact_input.mp4"
output_video = "test_artifact_output.mp4"
create_test_video(test_video)

# Create a simple mask (simulate a foreground object)
width, height = 640, 480
mask = np.zeros((height, width), dtype=np.uint8)
# Add a circular mask in the center
cv2.circle(mask, (width//2, height//2), 100, 255, -1)

# Simulate handoff data from TextBehindSegment
# Create a frozen text RGBA with "TEST" text
from PIL import Image, ImageDraw, ImageFont

text = "TEST"
font_size = 100
text_color = (255, 220, 0)

# Create the frozen text image
img = Image.new('RGBA', (width, height), (0, 0, 0, 0))
draw = ImageDraw.Draw(img)

# Try to load a font
try:
    font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", font_size)
except:
    font = ImageFont.load_default()

# Draw text with outline
text_bbox = draw.textbbox((0, 0), text, font=font)
text_width = text_bbox[2] - text_bbox[0]
text_height = text_bbox[3] - text_bbox[1]
text_x = (width - text_width) // 2
text_y = (height - text_height) // 2

# Draw outline
for dx in range(-2, 3):
    for dy in range(-2, 3):
        if abs(dx) == 2 or abs(dy) == 2:
            draw.text((text_x + dx, text_y + dy), text, font=font, fill=(255, 255, 255, 150))

# Draw main text
draw.text((text_x, text_y), text, font=font, fill=(*text_color, 255))

frozen_text_rgba = np.array(img)

# Create handoff data
handoff_data = {
    'final_text_rgba': frozen_text_rgba,
    'final_font_size': font_size,
    'text': text,
    'text_color': text_color,
    'final_occlusion': True,
    'outline_width': 2,
}

print(f"[TEST] Creating WordDissolve with text='{text}'")

# Create WordDissolve with the fix
word_dissolver = WordDissolve(
    element_path=test_video,
    background_path=test_video,
    position=(width//2, height//2),
    word=text,
    font_size=font_size,
    text_color=text_color,
    stable_duration=0.2,
    dissolve_duration=1.0,
    dissolve_stagger=0.2,
    float_distance=30,
    randomize_order=False,
    maintain_kerning=True,
    center_position=(width//2, height//2),
    handoff_data=handoff_data,
    fps=30,
    sprite_pad_ratio=0.3,  # Good padding to prevent artifacts
    debug=True
)

# Test rendering a few frames
print("\n[TEST] Testing dissolve frames...")

# Load video for background
cap = cv2.VideoCapture(test_video)

# Setup output video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video, fourcc, 30, (width, height))

# Process frames through dissolve
total_frames = 90  # 3 seconds
for frame_idx in range(total_frames):
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = cap.read()
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Apply the dissolve effect
    frame_rgb = word_dissolver.render_word_frame(frame_rgb, frame_idx, mask=mask)
    
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    out.write(frame_bgr)
    
    # Print progress for key frames
    if frame_idx == 0:
        print(f"[TEST] Frame {frame_idx}: Stable phase")
    elif frame_idx == word_dissolver.stable_frames:
        print(f"[TEST] Frame {frame_idx}: Dissolve begins")
    elif frame_idx % 15 == 0:
        print(f"[TEST] Frame {frame_idx}: Processing...")

cap.release()
out.release()

print(f"\n[TEST] Output saved to: {output_video}")
print("[TEST] Check the output video for rectangular artifacts around dissolving letters.")
print("[TEST] With the fix applied, letters should dissolve smoothly without visible bounding boxes.")