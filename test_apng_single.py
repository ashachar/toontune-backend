#!/usr/bin/env python3

import sys
import os
sys.path.append('video-processing')

from render_and_save_assets_apng import create_drawing_animation_apng, create_default_hand_image_with_alpha
import cv2

# Test with a single asset
asset_path = "../app/backend/uploads/assets/star.svg"

print(f"Testing APNG generation with: {asset_path}")

# Create default hand with alpha
hand_bgr, hand_alpha = create_default_hand_image_with_alpha()

# Scale hand
HAND_SCALE = 0.12
new_width = int(hand_bgr.shape[1] * HAND_SCALE)
new_height = int(hand_bgr.shape[0] * HAND_SCALE)
hand_bgr = cv2.resize(hand_bgr, (new_width, new_height))
hand_alpha = cv2.resize(hand_alpha, (new_width, new_height))

print(f"Hand shape: {hand_bgr.shape}, alpha shape: {hand_alpha.shape}")

# Process one asset
apng_path = create_drawing_animation_apng(asset_path, hand_bgr, hand_alpha)

if apng_path:
    print(f"Success! APNG created at: {apng_path}")
    # Check file size
    size = os.path.getsize(apng_path) / 1024
    print(f"APNG size: {size:.1f} KB")
    
    # Keep file for inspection
    import shutil
    output_path = "test_star_drawing.apng"
    shutil.copy(apng_path, output_path)
    print(f"Copied to: {output_path}")
    
    os.remove(apng_path)
else:
    print("Failed to create APNG")