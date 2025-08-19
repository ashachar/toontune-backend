#!/usr/bin/env python3
"""
Test script for direct pixel-based overlay positioning without grids.
Tests if LLM can directly determine pixel bounds for overlays.
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

from downsample.video_downsample import VideoDownsampler


def extract_frame_at_timestamp(video_path: str, timestamp: float) -> np.ndarray:
    """Extract a frame at the given timestamp from the video."""
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_number = int(timestamp * fps)
    
    # Clamp frame number to valid range
    frame_number = min(frame_number, total_frames - 1)
    frame_number = max(0, frame_number)
    
    # Set position to the frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    
    # Read the frame
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        raise ValueError(f"Could not extract frame at timestamp {timestamp}")
    
    return frame


def create_pixel_positioning_prompt(
    placement_description: str,
    word: str,
    element_width: int,
    element_height: int,
    frame_width: int,
    frame_height: int,
    interaction_style: str
) -> str:
    """Create a prompt for direct pixel positioning."""
    
    prompt = f"""You are analyzing a video frame to determine where to place a text overlay.

Frame dimensions: {frame_width}x{frame_height} pixels
Text overlay: "{word}"
Required overlay size: {element_width}x{element_height} pixels
Placement instruction: {placement_description}
Interaction style: {interaction_style}

The coordinate system:
- Origin (0,0) is at the top-left corner
- X increases rightward (0 to {frame_width-1})
- Y increases downward (0 to {frame_height-1})

Based on the placement instruction and the visual content of the frame, determine the exact pixel bounds for the overlay.

Return ONLY a JSON object with the pixel coordinates:
{{
  "top_left": {{
    "x": <x_coordinate>,
    "y": <y_coordinate>
  }},
  "bottom_right": {{
    "x": <x_coordinate>,
    "y": <y_coordinate>
  }}
}}

The bounding box should:
1. Be exactly {element_width} pixels wide and {element_height} pixels tall
2. Follow the placement instruction "{placement_description}"
3. Not exceed frame boundaries (0 to {frame_width-1} for x, 0 to {frame_height-1} for y)
4. Be positioned for optimal visual balance and readability

Example valid response for a 100x50 overlay in top-left corner:
{{
  "top_left": {{
    "x": 20,
    "y": 20
  }},
  "bottom_right": {{
    "x": 119,
    "y": 69
  }}
}}

Important: bottom_right.x = top_left.x + {element_width - 1}, bottom_right.y = top_left.y + {element_height - 1}"""
    
    return prompt


def visualize_overlay_bounds(
    frame: np.ndarray,
    top_left: Tuple[int, int],
    bottom_right: Tuple[int, int],
    word: str,
    output_path: str
) -> None:
    """Draw the overlay bounds on the frame and save it."""
    
    # Create a copy to draw on
    vis_frame = frame.copy()
    
    # Draw the bounding box
    cv2.rectangle(vis_frame, top_left, bottom_right, (0, 255, 0), 2)
    
    # Draw corner markers
    marker_size = 10
    # Top-left
    cv2.line(vis_frame, top_left, (top_left[0] + marker_size, top_left[1]), (0, 255, 255), 3)
    cv2.line(vis_frame, top_left, (top_left[0], top_left[1] + marker_size), (0, 255, 255), 3)
    # Bottom-right
    cv2.line(vis_frame, bottom_right, (bottom_right[0] - marker_size, bottom_right[1]), (0, 255, 255), 3)
    cv2.line(vis_frame, bottom_right, (bottom_right[0], bottom_right[1] - marker_size), (0, 255, 255), 3)
    
    # Add text label
    cv2.putText(vis_frame, f'"{word}"', 
                (top_left[0], top_left[1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    # Add dimensions
    width = bottom_right[0] - top_left[0] + 1
    height = bottom_right[1] - top_left[1] + 1
    dim_text = f"{width}x{height}px"
    cv2.putText(vis_frame, dim_text,
                (top_left[0], bottom_right[1] + 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
    
    # Save the visualization
    cv2.imwrite(output_path, vis_frame)
    print(f"Visualization saved to: {output_path}")


def test_single_overlay():
    """Test pixel positioning on a single frame."""
    
    # Test case: "Let's" at 2.779s, should be in top-left corner
    video_path = "uploads/assets/videos/do_re_mi_with_music/scenes/scene_001.mp4"
    timestamp = 2.779
    word = "Let's"
    placement = "top left corner"
    element_width = 100
    element_height = 50
    interaction_style = "anchored_to_background"
    
    print("=" * 60)
    print("PIXEL-BASED OVERLAY POSITIONING TEST")
    print("=" * 60)
    print(f"Testing: '{word}' at {timestamp}s")
    print(f"Placement: {placement}")
    print(f"Element size: {element_width}x{element_height}")
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
    
    # Simulate LLM response (in production, this would be an API call)
    # For "top left corner", a reasonable position would be around (20-50, 20-50)
    simulated_response = {
        "top_left": {
            "x": 30,
            "y": 25
        },
        "bottom_right": {
            "x": 129,  # 30 + 100 - 1
            "y": 74    # 25 + 50 - 1
        }
    }
    
    print("\nSimulated LLM Response:")
    print(json.dumps(simulated_response, indent=2))
    
    # Validate response
    top_left = (simulated_response["top_left"]["x"], simulated_response["top_left"]["y"])
    bottom_right = (simulated_response["bottom_right"]["x"], simulated_response["bottom_right"]["y"])
    
    # Check dimensions
    actual_width = bottom_right[0] - top_left[0] + 1
    actual_height = bottom_right[1] - top_left[1] + 1
    
    print("\nValidation:")
    print(f"  Expected size: {element_width}x{element_height}")
    print(f"  Actual size: {actual_width}x{actual_height}")
    print(f"  Size match: {'✓' if actual_width == element_width and actual_height == element_height else '✗'}")
    
    # Check bounds
    in_bounds = (0 <= top_left[0] < frame_width and 
                 0 <= top_left[1] < frame_height and
                 0 <= bottom_right[0] < frame_width and
                 0 <= bottom_right[1] < frame_height)
    print(f"  Within frame bounds: {'✓' if in_bounds else '✗'}")
    
    # Check placement logic
    is_top_left = top_left[0] < frame_width * 0.3 and top_left[1] < frame_height * 0.3
    print(f"  In top-left region: {'✓' if is_top_left else '✗'}")
    
    # Visualize
    vis_path = output_dir / f"visualization_{word}.png"
    visualize_overlay_bounds(frame, top_left, bottom_right, word, str(vis_path))
    
    print("\n" + "=" * 60)
    print("ANALYSIS")
    print("=" * 60)
    
    if actual_width == element_width and actual_height == element_height and in_bounds and is_top_left:
        print("✓ SUCCESS: The positioning makes sense!")
        print(f"  - Overlay is correctly sized ({element_width}x{element_height})")
        print(f"  - Position ({top_left[0]}, {top_left[1]}) is in the top-left corner")
        print("  - All coordinates are within frame boundaries")
    else:
        print("✗ ISSUES DETECTED:")
        if actual_width != element_width or actual_height != element_height:
            print(f"  - Size mismatch: Expected {element_width}x{element_height}, got {actual_width}x{actual_height}")
            print("    Likely cause: Math error in calculating bottom_right from top_left")
        if not in_bounds:
            print("  - Coordinates exceed frame boundaries")
            print("    Likely cause: LLM didn't properly constrain to frame dimensions")
        if not is_top_left:
            print("  - Position is not in top-left corner as requested")
            print("    Likely cause: LLM misunderstood the placement instruction")
    
    print("\nPotential improvements for the prompt:")
    print("1. Add more explicit examples for different placements")
    print("2. Emphasize the math: bottom_right = top_left + size - 1")
    print("3. Provide visual zones (e.g., top-left = x<30% of width, y<30% of height)")
    print("4. Include a 'common mistakes to avoid' section")
    
    return output_dir


if __name__ == "__main__":
    output_dir = test_single_overlay()
    print(f"\nAll test files saved in: {output_dir}/")
    print("\nTo view the results:")
    print(f"open {output_dir}/*.png")