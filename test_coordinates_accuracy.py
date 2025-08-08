#!/usr/bin/env python3

import os
import sys
import json
import tempfile

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from render_and_save_assets_apng import (
    create_drawing_animation_apng,
    load_hand_image,
    HAND_IMAGE_PATH,
    HAND_SCALE
)
import cv2

def test_coordinates_accuracy():
    """Test that hand coordinates are accurate and properly positioned"""
    
    print("=" * 60)
    print("Testing Hand Coordinates Accuracy")
    print("=" * 60)
    
    # Use woman.png as test asset
    test_asset = "../app/uploads/assets/woman.png"
    
    if not os.path.exists(test_asset):
        print(f"Error: Test asset not found: {test_asset}")
        return False
    
    print(f"\nTest asset: {test_asset}")
    
    # Load and check asset dimensions
    img = cv2.imread(test_asset, cv2.IMREAD_UNCHANGED)
    if img is None:
        print("Error: Could not load test image")
        return False
    
    height, width = img.shape[:2]
    print(f"Asset dimensions: {width}x{height}")
    
    # Load hand image
    hand_bgr, hand_alpha = load_hand_image(HAND_IMAGE_PATH)
    if hand_bgr is None:
        print("Error: Could not load hand image")
        return False
    
    # Scale hand
    new_width = int(hand_bgr.shape[1] * HAND_SCALE)
    new_height = int(hand_bgr.shape[0] * HAND_SCALE)
    hand_bgr = cv2.resize(hand_bgr, (new_width, new_height))
    if hand_alpha is not None:
        hand_alpha = cv2.resize(hand_alpha, (new_width, new_height))
    
    print(f"Hand dimensions after scaling: {new_width}x{new_height}")
    print(f"Hand scale factor: {HAND_SCALE}")
    
    # Generate APNG and coordinates
    print("\nGenerating APNG and hand coordinates...")
    result = create_drawing_animation_apng(test_asset, hand_bgr, hand_alpha, verbose=False)
    
    if not result:
        print("Error: Failed to generate APNG")
        return False
    
    apng_path, hand_coords = result
    
    print(f"\nGenerated {len(hand_coords)} hand coordinate frames")
    
    # Analyze coordinates
    visible_coords = [c for c in hand_coords if c['visible']]
    hidden_coords = [c for c in hand_coords if not c['visible']]
    
    print(f"Visible frames: {len(visible_coords)}")
    print(f"Hidden frames: {len(hidden_coords)}")
    
    if visible_coords:
        # Find coordinate ranges
        x_coords = [c['x'] for c in visible_coords]
        y_coords = [c['y'] for c in visible_coords]
        
        print(f"\nCoordinate ranges:")
        print(f"  X: {min(x_coords)} to {max(x_coords)} (range: {max(x_coords) - min(x_coords)})")
        print(f"  Y: {min(y_coords)} to {max(y_coords)} (range: {max(y_coords) - min(y_coords)})")
        
        # Check if coordinates are within image bounds
        # Hand coordinates should account for the pen offset
        pen_offset_x = new_width // 3
        pen_offset_y = new_height // 3
        
        print(f"\nPen offset: ({pen_offset_x}, {pen_offset_y})")
        
        # Adjusted bounds check
        out_of_bounds = 0
        for c in visible_coords:
            # The actual drawing point is at (x + pen_offset_x, y + pen_offset_y)
            draw_x = c['x'] + pen_offset_x
            draw_y = c['y'] + pen_offset_y
            
            if draw_x < 0 or draw_x >= width or draw_y < 0 or draw_y >= height:
                out_of_bounds += 1
        
        if out_of_bounds > 0:
            print(f"WARNING: {out_of_bounds} coordinates would place the pen tip outside image bounds")
        else:
            print("✓ All pen tip positions are within image bounds")
        
        # Sample some coordinates for visual verification
        print(f"\nSample coordinates (first 10 visible frames):")
        for i, coord in enumerate(visible_coords[:10]):
            draw_x = coord['x'] + pen_offset_x
            draw_y = coord['y'] + pen_offset_y
            print(f"  Frame {i}: Hand=({coord['x']}, {coord['y']}), Pen tip=({draw_x}, {draw_y})")
    
    # Clean up
    try:
        os.remove(apng_path)
    except:
        pass
    
    print("\n" + "=" * 60)
    print("COORDINATE TEST COMPLETE")
    print("=" * 60)
    
    return True

if __name__ == "__main__":
    success = test_coordinates_accuracy()
    sys.exit(0 if success else 1)