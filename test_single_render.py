#!/usr/bin/env python3

import sys
import os
sys.path.append('video-processing')

from render_and_save_assets_offline import create_drawing_animation_for_asset, create_default_hand_image

# Test with a single asset
asset_path = "../app/backend/uploads/assets/star.svg"

print(f"Testing with: {asset_path}")

# Create default hand
hand_img, hand_alpha = create_default_hand_image()

# Scale hand
import cv2
HAND_SCALE = 0.12
new_width = int(hand_img.shape[1] * HAND_SCALE)
new_height = int(hand_img.shape[0] * HAND_SCALE)
hand_img = cv2.resize(hand_img, (new_width, new_height))

# Process one asset
video_path = create_drawing_animation_for_asset(asset_path, hand_img, hand_alpha)

if video_path:
    print(f"Success! Video created at: {video_path}")
    # Check file size
    size = os.path.getsize(video_path) / 1024
    print(f"Video size: {size:.1f} KB")
    os.remove(video_path)
else:
    print("Failed to create video")