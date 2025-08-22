#!/usr/bin/env python3
"""Quick test to verify dissolve fixes - letters disappearing and no frame artifacts."""

import os
os.environ['FRAME_DISSOLVE_DEBUG'] = '1'

import cv2
import numpy as np
from utils.animations.word_dissolve import WordDissolve
from PIL import Image, ImageDraw, ImageFont

# Create simple test video
def create_test_video(filename, frames=30):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, 30, (640, 480))
    for _ in range(frames):
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 100
        out.write(frame)
    out.release()
    return filename

# Create test
test_video = create_test_video("quick_test.mp4")
output_video = "quick_dissolve_output.mp4"

# Create frozen text
text = "ABC"
width, height = 640, 480
img = Image.new('RGBA', (width, height), (0, 0, 0, 0))
draw = ImageDraw.Draw(img)

try:
    font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 120)
except:
    font = ImageFont.load_default()

# Center text
bbox = draw.textbbox((0, 0), text, font=font)
text_width = bbox[2] - bbox[0]
text_height = bbox[3] - bbox[1]
text_x = (width - text_width) // 2
text_y = (height - text_height) // 2

# Draw with outline
for dx in range(-2, 3):
    for dy in range(-2, 3):
        if abs(dx) == 2 or abs(dy) == 2:
            draw.text((text_x + dx, text_y + dy), text, font=font, fill=(255, 255, 255, 150))
draw.text((text_x, text_y), text, font=font, fill=(255, 220, 0, 255))

frozen_text_rgba = np.array(img)

# Create WordDissolve with short durations for quick test
word_dissolver = WordDissolve(
    element_path=test_video,
    background_path=test_video,
    position=(width//2, height//2),
    word=text,
    font_size=120,
    text_color=(255, 220, 0),
    stable_duration=0.1,      # 3 frames
    dissolve_duration=0.5,     # 15 frames
    dissolve_stagger=0.1,      # 3 frames between letters
    float_distance=20,
    randomize_order=False,
    maintain_kerning=True,
    center_position=(width//2, height//2),
    handoff_data={
        'final_text_rgba': frozen_text_rgba,
        'final_font_size': 120,
        'text': text,
        'text_color': (255, 220, 0),
        'final_occlusion': False,
        'outline_width': 2,
    },
    fps=30,
    sprite_pad_ratio=0.40,  # High padding for safety
    debug=True
)

print(f"\n[TEST] Quick dissolve test with text='{text}'")
print(f"[TEST] Stable: {word_dissolver.stable_frames}f, Dissolve: {word_dissolver.dissolve_frames}f, Stagger: {word_dissolver.stagger_frames}f")

# Process frames
cap = cv2.VideoCapture(test_video)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video, fourcc, 30, (width, height))

# Test critical frames
critical_frames = [
    0,   # Stable
    3,   # First dissolve starts
    6,   # Second dissolve starts  
    9,   # Third dissolve starts
    12,  # Mid dissolve
    15,  # Late dissolve
    18,  # First should be done
    21,  # Second should be done
    24,  # Third should be done
    27,  # All should be gone
    30,  # Verify all gone
]

total_frames = 35
for frame_idx in range(total_frames):
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = cap.read()
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_rgb = word_dissolver.render_word_frame(frame_rgb, frame_idx, mask=None)
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    out.write(frame_bgr)
    
    if frame_idx in critical_frames:
        print(f"\n[CRITICAL] Frame {frame_idx} processed")

cap.release()
out.release()

# Convert to H.264
import subprocess
cmd = [
    'ffmpeg', '-y', '-i', output_video,
    '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
    '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
    output_video.replace('.mp4', '_h264.mp4')
]
subprocess.run(cmd, capture_output=True, text=True)

print(f"\n[TEST] Output: {output_video.replace('.mp4', '_h264.mp4')}")
print("[TEST] Check frames 18+ to verify letters are GONE, not returning")
print("[TEST] Check for any rectangular artifacts around dissolving letters")