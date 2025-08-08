#!/usr/bin/env python3

import sys
import os
import cv2
sys.path.append('video-processing')

from render_and_save_assets_apng import create_drawing_animation_apng
from generate_drawing_video import load_hand_image

# Test with a single asset
asset_path = "../app/uploads/assets/star.svg"

print(f"Testing with larger hand (50% bigger)")
print(f"Asset: {asset_path}")

# Load hand with new scale
hand_path = "../app/uploads/assets/hand.png"
hand_bgr, hand_alpha = load_hand_image(hand_path)

if hand_bgr is None:
    print(f"Error: Could not load hand from {hand_path}")
    sys.exit(1)

# Apply new scale
HAND_SCALE = 0.225  # 50% larger
new_width = int(hand_bgr.shape[1] * HAND_SCALE)
new_height = int(hand_bgr.shape[0] * HAND_SCALE)
hand_bgr = cv2.resize(hand_bgr, (new_width, new_height))
if hand_alpha is not None:
    hand_alpha = cv2.resize(hand_alpha, (new_width, new_height))

print(f"Hand original size: 1024x1024")
print(f"Hand scaled size: {new_width}x{new_height} pixels")

# Process one asset with verbose mode
apng_path = create_drawing_animation_apng(asset_path, hand_bgr, hand_alpha, verbose=True)

if apng_path:
    print(f"\nSuccess! APNG created at: {apng_path}")
    size = os.path.getsize(apng_path) / 1024
    print(f"APNG size: {size:.1f} KB")
    
    # Save for inspection
    import shutil
    output_path = "test_star_larger_hand.apng"
    shutil.copy(apng_path, output_path)
    print(f"Copied to: {output_path}")
    
    os.remove(apng_path)
else:
    print("Failed to create APNG")