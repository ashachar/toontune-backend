#!/usr/bin/env python3
"""
Test more complex pixel positioning - avoiding subject occlusion.
"""

import cv2
import numpy as np
import json
import os
import sys
from pathlib import Path
import tempfile
from typing import Dict, Tuple

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from test_pixel_positioning import (
    extract_frame_at_timestamp,
    create_pixel_positioning_prompt,
    visualize_overlay_bounds
)


def test_complex_overlay():
    """Test pixel positioning for 'sun' which should be in sky above the woman."""
    
    # Test case: "sun" at 46.34s, should be in sky above her
    video_path = "uploads/assets/videos/do_re_mi_with_music/scenes/scene_001.mp4"
    timestamp = 46.34
    word = "sun"
    placement = "in the sky above her"
    element_width = 120
    element_height = 60
    interaction_style = "anchored_to_background"
    
    print("=" * 60)
    print("COMPLEX PIXEL POSITIONING TEST")
    print("=" * 60)
    print(f"Testing: '{word}' at {timestamp}s")
    print(f"Placement: {placement}")
    print(f"Element size: {element_width}x{element_height}")
    print("Challenge: Must be in sky area, avoiding the woman's face/body")
    print()
    
    # Extract frame
    print("Extracting frame...")
    frame = extract_frame_at_timestamp(video_path, timestamp)
    frame_height, frame_width = frame.shape[:2]
    print(f"Frame dimensions: {frame_width}x{frame_height}")
    
    # Save original frame for reference
    output_dir = Path("utils/video_overlay/pixel_test")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    original_path = output_dir / f"original_frame_{word}.png"
    cv2.imwrite(str(original_path), frame)
    print(f"Original frame saved to: {original_path}")
    
    # Create prompt
    prompt = create_pixel_positioning_prompt(
        placement,
        word,
        element_width,
        element_height,
        frame_width,
        frame_height,
        interaction_style
    )
    
    # Save prompt
    prompt_path = output_dir / f"prompt_{word}.txt"
    with open(prompt_path, 'w') as f:
        f.write(prompt)
    print(f"Prompt saved to: {prompt_path}")
    
    print("\n" + "-" * 40)
    print("TESTING MULTIPLE POSITIONING ATTEMPTS")
    print("-" * 40)
    
    # Test multiple simulated responses to see which makes sense
    test_positions = [
        {
            "name": "Top-right sky",
            "response": {
                "top_left": {"x": 120, "y": 10},
                "bottom_right": {"x": 239, "y": 69}
            }
        },
        {
            "name": "Top-center sky",
            "response": {
                "top_left": {"x": 68, "y": 5},
                "bottom_right": {"x": 187, "y": 64}
            }
        },
        {
            "name": "Above head (might overlap)",
            "response": {
                "top_left": {"x": 80, "y": 25},
                "bottom_right": {"x": 199, "y": 84}
            }
        }
    ]
    
    for i, test in enumerate(test_positions, 1):
        print(f"\nTest {i}: {test['name']}")
        response = test['response']
        print(f"  Position: ({response['top_left']['x']}, {response['top_left']['y']})")
        
        # Validate response
        top_left = (response["top_left"]["x"], response["top_left"]["y"])
        bottom_right = (response["bottom_right"]["x"], response["bottom_right"]["y"])
        
        # Check dimensions
        actual_width = bottom_right[0] - top_left[0] + 1
        actual_height = bottom_right[1] - top_left[1] + 1
        
        # Check bounds
        in_bounds = (0 <= top_left[0] <= frame_width - element_width and 
                     0 <= top_left[1] <= frame_height - element_height and
                     bottom_right[0] < frame_width and
                     bottom_right[1] < frame_height)
        
        # Check if in upper portion (sky area)
        in_sky = top_left[1] < frame_height * 0.4  # Top 40% of frame
        
        print(f"  Size: {actual_width}x{actual_height} (Expected: {element_width}x{element_height}) {'✓' if actual_width == element_width and actual_height == element_height else '✗'}")
        print(f"  Within bounds: {'✓' if in_bounds else '✗'}")
        print(f"  In sky area (top 40%): {'✓' if in_sky else '✗'}")
        
        # Visualize
        vis_path = output_dir / f"visualization_{word}_test{i}.png"
        visualize_overlay_bounds(frame, top_left, bottom_right, f"{word} ({test['name']})", str(vis_path))
    
    print("\n" + "=" * 60)
    print("ANALYSIS")
    print("=" * 60)
    
    print("\nWhat makes pixel-based positioning challenging:")
    print("1. LLM needs to 'see' the image to avoid occlusion")
    print("2. Must understand spatial relationships (what is 'sky', 'above her')")
    print("3. Needs to calculate exact pixel math correctly")
    print("4. Balance between following instruction and visual aesthetics")
    
    print("\nPros of pixel-based approach:")
    print("✓ More precise control over exact positioning")
    print("✓ No grid quantization - can place anywhere")
    print("✓ Simpler system - no intermediate grid layer")
    
    print("\nCons of pixel-based approach:")
    print("✗ Harder for LLM to reason about positions")
    print("✗ More prone to calculation errors")
    print("✗ Difficult to ensure consistent spacing without grid reference")
    print("✗ LLM might place overlays in visually poor locations")
    
    print("\nRecommendation:")
    print("The GRID approach is likely better because:")
    print("- Provides visual reference points for the LLM")
    print("- Reduces calculation errors")
    print("- Easier to describe relative positions")
    print("- Grid cells act as 'safe zones' for placement")
    
    return output_dir


if __name__ == "__main__":
    output_dir = test_complex_overlay()
    print(f"\nAll test files saved in: {output_dir}/")
    print("\nTo view all visualizations:")
    print(f"open {output_dir}/visualization_sun*.png")