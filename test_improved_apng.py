#!/usr/bin/env python3

import sys
import os
sys.path.append('video-processing')

from render_and_save_assets_apng import create_drawing_animation_apng, create_default_hand_image_with_alpha
import cv2

# Test with a single asset
asset_path = "../app/backend/uploads/assets/star.svg"

print(f"Testing improved APNG generation with: {asset_path}")

# Create default hand with alpha (now larger)
hand_bgr, hand_alpha = create_default_hand_image_with_alpha()

# Scale hand with new HAND_SCALE
HAND_SCALE = 0.5  # Updated scale
new_width = int(hand_bgr.shape[1] * HAND_SCALE)
new_height = int(hand_bgr.shape[0] * HAND_SCALE)
hand_bgr = cv2.resize(hand_bgr, (new_width, new_height))
hand_alpha = cv2.resize(hand_alpha, (new_width, new_height))

print(f"Hand shape: {hand_bgr.shape} (scaled from 300x225 with scale {HAND_SCALE})")
print(f"Final hand size: {new_width}x{new_height} pixels")

# Process one asset with verbose mode to see touch-up phase
apng_path = create_drawing_animation_apng(asset_path, hand_bgr, hand_alpha, verbose=True)

if apng_path:
    print(f"\nSuccess! APNG created at: {apng_path}")
    # Check file size
    size = os.path.getsize(apng_path) / 1024
    print(f"APNG size: {size:.1f} KB")
    
    # Keep file for inspection
    import shutil
    output_path = "test_star_improved.apng"
    shutil.copy(apng_path, output_path)
    print(f"Copied to: {output_path} for inspection")
    
    os.remove(apng_path)
else:
    print("Failed to create APNG")