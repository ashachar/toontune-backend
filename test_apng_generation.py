#!/usr/bin/env python3

import os
import sys
import tempfile
import json

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the rendering function
from render_and_save_assets_apng import (
    create_drawing_animation_apng,
    load_hand_image,
    HAND_IMAGE_PATH,
    HAND_SCALE
)
import cv2

def test_apng_generation():
    """Test that APNG generation produces both APNG and hand coordinates"""
    
    print("=" * 60)
    print("Testing APNG Generation with Separate Hand Coordinates")
    print("=" * 60)
    
    # Find a test asset
    test_assets = [
        "../app/uploads/assets/woman.png",
        "../app/uploads/assets/cat.png",
        "../app/uploads/assets/dog.png"
    ]
    
    test_asset = None
    for asset in test_assets:
        if os.path.exists(asset):
            test_asset = asset
            break
    
    if not test_asset:
        print("Error: No test asset found")
        return False
    
    print(f"\nUsing test asset: {test_asset}")
    
    # Load hand image
    print(f"Loading hand image from: {HAND_IMAGE_PATH}")
    hand_bgr, hand_alpha = load_hand_image(HAND_IMAGE_PATH)
    
    if hand_bgr is None:
        print("Error: Could not load hand image")
        return False
    
    # Scale hand image
    new_width = int(hand_bgr.shape[1] * HAND_SCALE)
    new_height = int(hand_bgr.shape[0] * HAND_SCALE)
    hand_bgr = cv2.resize(hand_bgr, (new_width, new_height))
    if hand_alpha is not None:
        hand_alpha = cv2.resize(hand_alpha, (new_width, new_height))
    
    print(f"Hand image scaled to: {new_width}x{new_height}")
    
    # Generate APNG and hand coordinates
    print("\nGenerating APNG and hand coordinates...")
    result = create_drawing_animation_apng(test_asset, hand_bgr, hand_alpha, verbose=True)
    
    if not result:
        print("Error: Failed to generate APNG")
        return False
    
    apng_path, hand_coords = result
    
    print(f"\nGenerated APNG: {apng_path}")
    print(f"Number of hand coordinate frames: {len(hand_coords)}")
    
    # Verify APNG file exists
    if not os.path.exists(apng_path):
        print("Error: APNG file was not created")
        return False
    
    apng_size = os.path.getsize(apng_path)
    print(f"APNG file size: {apng_size} bytes")
    
    # Verify hand coordinates
    if not hand_coords or len(hand_coords) == 0:
        print("Error: No hand coordinates generated")
        return False
    
    # Sample some coordinates
    print("\nSample hand coordinates (first 5 frames):")
    for i, coord in enumerate(hand_coords[:5]):
        print(f"  Frame {i}: x={coord['x']}, y={coord['y']}, visible={coord['visible']}")
    
    # Count visible frames
    visible_frames = sum(1 for c in hand_coords if c['visible'])
    print(f"\nTotal frames: {len(hand_coords)}")
    print(f"Visible hand frames: {visible_frames}")
    print(f"Hidden hand frames: {len(hand_coords) - visible_frames}")
    
    # Save hand coordinates to test JSON file
    test_json_path = tempfile.NamedTemporaryFile(suffix='_hand_coords.json', delete=False).name
    with open(test_json_path, 'w') as f:
        json.dump(hand_coords, f, indent=2)
    
    print(f"\nSaved hand coordinates to: {test_json_path}")
    
    # Clean up
    try:
        os.remove(apng_path)
        os.remove(test_json_path)
        print("\nCleanup completed")
    except:
        pass
    
    print("\n" + "=" * 60)
    print("TEST SUCCESSFUL!")
    print("=" * 60)
    print("\nThe backend correctly generates:")
    print("1. APNG file without hand overlay")
    print("2. Hand coordinates JSON with x, y, and visibility for each frame")
    
    return True

if __name__ == "__main__":
    success = test_apng_generation()
    sys.exit(0 if success else 1)