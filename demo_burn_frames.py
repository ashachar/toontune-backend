#!/usr/bin/env python3
"""
Generate key frames showing the photorealistic burn effect.
"""

import cv2
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.animations.letter_3d_burn.photorealistic_burn import PhotorealisticLetterBurn

print("="*60)
print("PHOTOREALISTIC BURN - KEY FRAMES DEMO")
print("="*60)

# Create burn animation
burn = PhotorealisticLetterBurn(
    duration=4.0,
    fps=30,
    resolution=(1280, 720),
    text="BURN",
    font_size=200,
    text_color=(200, 200, 200),
    flame_height=150,
    burn_stagger=0.25,
    supersample_factor=1
)

# Dark background
background = np.ones((720, 1280, 3), dtype=np.uint8) * 30

# Key frames to generate
key_frames = [
    (0, "Initial State"),
    (20, "Starting to Ignite"),
    (40, "First Letter Burning"),
    (60, "Fire Spreading"),
    (80, "Intense Burning"),
    (100, "Heavy Smoke")
]

os.makedirs("outputs/burn_frames", exist_ok=True)

# Create montage
montage_width = 1280 * 2
montage_height = 720 * 3
montage = np.zeros((montage_height, montage_width, 3), dtype=np.uint8)

print("\nGenerating key frames...")

for idx, (frame_num, description) in enumerate(key_frames):
    print(f"  Frame {frame_num}: {description}")
    
    # Generate frame
    frame = burn.generate_frame(frame_num, background.copy())
    
    # Add label
    label_bg = np.zeros((50, 1280, 3), dtype=np.uint8)
    cv2.putText(label_bg, f"Frame {frame_num}: {description}",
               (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2)
    
    # Combine frame with label
    labeled_frame = np.vstack([label_bg, frame])
    
    # Save individual frame
    cv2.imwrite(f"outputs/burn_frames/frame_{frame_num:03d}.png", labeled_frame)
    
    # Add to montage
    row = idx // 2
    col = idx % 2
    y_start = row * (720 + 50)
    x_start = col * 1280
    montage[y_start:y_start + 770, x_start:x_start + 1280] = labeled_frame

# Save montage
cv2.imwrite("outputs/photorealistic_burn_montage.png", montage)

print("\n✅ Results saved:")
print("  • Individual frames: outputs/burn_frames/")
print("  • Montage: outputs/photorealistic_burn_montage.png")
print("\n✨ Features demonstrated:")
print("  • Realistic fire with Perlin noise")
print("  • Volumetric smoke rendering")
print("  • Progressive material burning")
print("  • Heat propagation effects")
print("  • Glowing embers and charring")