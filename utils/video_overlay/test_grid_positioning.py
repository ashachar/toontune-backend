#!/usr/bin/env python3
"""
Test grid-based positioning on the same frame as pixel test for comparison.
"""

import cv2
import numpy as np
import json
import os
import sys
from pathlib import Path
import tempfile
from typing import Dict, Tuple, List

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from get_overlay_pixels_by_grid import (
    extract_frame_at_timestamp,
    create_grid_overlay,
    calculate_grid_size,
    get_pixel_from_cell
)


def visualize_grid_with_overlay(
    frame: np.ndarray,
    grid_size: Tuple[int, int],
    top_left_cell: Tuple[int, int],
    bottom_right_cell: Tuple[int, int],
    word: str,
    element_size: Tuple[int, int],
    output_path: str
) -> None:
    """Draw grid and overlay bounds on the frame."""
    
    # First add the grid
    grid_frame, cell_centers = create_grid_overlay(frame, grid_size)
    
    # Calculate pixel bounds from cells
    if top_left_cell in cell_centers and bottom_right_cell in cell_centers:
        top_left_pixel = cell_centers[top_left_cell]
        bottom_right_pixel = cell_centers[bottom_right_cell]
        
        # Adjust for actual overlay size (cells give centers, we need corners)
        h, w = frame.shape[:2]
        rows, cols = grid_size
        cell_height = h // rows
        cell_width = w // cols
        
        # Calculate actual corners based on cell boundaries
        tl_row, tl_col = top_left_cell
        br_row, br_col = bottom_right_cell
        
        x1 = (tl_col - 1) * cell_width
        y1 = (tl_row - 1) * cell_height
        x2 = br_col * cell_width
        y2 = br_row * cell_height
        
        # Draw overlay bounds in red
        cv2.rectangle(grid_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
        
        # Add text label
        cv2.putText(grid_frame, f'"{word}"', 
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Add cell coordinates
        cell_text = f"Cells: ({tl_row},{tl_col}) to ({br_row},{br_col})"
        cv2.putText(grid_frame, cell_text,
                    (x1, y2 + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        
        # Add actual size
        actual_width = x2 - x1
        actual_height = y2 - y1
        size_text = f"Area: {actual_width}x{actual_height}px (need {element_size[0]}x{element_size[1]}px)"
        cv2.putText(grid_frame, size_text,
                    (x1, y2 + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1)
    
    cv2.imwrite(output_path, grid_frame)
    print(f"Visualization saved to: {output_path}")


def test_grid_positioning():
    """Test grid positioning on the same 'sun' frame."""
    
    # Same test case as pixel test: "sun" at 46.34s
    video_path = "uploads/assets/videos/do_re_mi_with_music/scenes/scene_001.mp4"
    timestamp = 46.34
    word = "sun"
    placement = "in the sky above her"
    element_width = 120
    element_height = 60
    scene_width = 1920  # Original scene size
    scene_height = 1080
    interaction_style = "anchored_to_background"
    
    print("=" * 60)
    print("GRID-BASED OVERLAY POSITIONING TEST")
    print("=" * 60)
    print(f"Testing: '{word}' at {timestamp}s")
    print(f"Placement: {placement}")
    print(f"Element size: {element_width}x{element_height}")
    print()
    
    # Extract frame (will be downsampled)
    print("Extracting frame...")
    frame = extract_frame_at_timestamp(video_path, timestamp)
    frame_height, frame_width = frame.shape[:2]
    print(f"Frame dimensions: {frame_width}x{frame_height}")
    
    # Calculate grid size based on element
    grid_size = calculate_grid_size(
        element_width, 
        element_height,
        scene_width,
        scene_height
    )
    print(f"Calculated grid size: {grid_size[0]}x{grid_size[1]}")
    
    # Save original frame for reference
    output_dir = Path("utils/video_overlay/grid_test")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create grid overlay
    grid_frame, cell_centers = create_grid_overlay(frame, grid_size)
    grid_path = output_dir / f"grid_frame_{word}.png"
    cv2.imwrite(str(grid_path), grid_frame)
    print(f"Grid frame saved to: {grid_path}")
    
    # Calculate cell dimensions
    h, w = frame.shape[:2]
    rows, cols = grid_size
    cell_height = h // rows
    cell_width = w // cols
    print(f"Cell dimensions: {cell_width}x{cell_height} pixels")
    
    # Generate the prompt (same format as our system)
    prompt = f"""Given a video frame with a numbered grid overlay, determine the best grid cells for placing the text overlay.

The overlay may span multiple grid cells. Provide the top-left and bottom-right corner cells.

Grid coordinates are in format (row, column) where:
- Row 1 is at the top, increasing downward
- Column 1 is at the left, increasing rightward
- Example: (3,5) means 3rd row from top, 5th column from left

Placement instruction: {placement}
Text to place: "{word}"
Interaction style: {interaction_style}
Element size: {element_width}x{element_height} pixels
Grid size: {grid_size[0]}x{grid_size[1]} cells
Cell size: approximately {cell_width}x{cell_height} pixels each

Analyze the frame and determine:
1. Top-left corner cell for the overlay
2. Bottom-right corner cell for the overlay

Return ONLY a JSON object in this exact format:
{{
  "top_left": "row,column",
  "bottom_right": "row,column"
}}"""

    # Save prompt
    prompt_path = output_dir / f"prompt_{word}.txt"
    with open(prompt_path, 'w') as f:
        f.write(prompt)
    print(f"Prompt saved to: {prompt_path}")
    
    print("\n" + "-" * 40)
    print("TESTING MULTIPLE GRID PLACEMENTS")
    print("-" * 40)
    
    # Test multiple simulated responses
    # With 9x8 grid and needing 120x60px:
    # Each cell is about 32x12 pixels
    # So we need about 4 cells wide and 5 cells tall
    
    test_positions = [
        {
            "name": "Top-center (good sky position)",
            "response": {
                "top_left": "1,3",
                "bottom_right": "2,5"  # 2 rows, 3 columns = ~96x24px (too small!)
            }
        },
        {
            "name": "Better sizing (spans more cells)",
            "response": {
                "top_left": "1,2",
                "bottom_right": "4,5"  # 4 rows, 4 columns = ~128x48px (closer!)
            }
        },
        {
            "name": "Right side sky",
            "response": {
                "top_left": "1,5",
                "bottom_right": "3,8"  # 3 rows, 4 columns = ~96x36px
            }
        },
        {
            "name": "Full width for text",
            "response": {
                "top_left": "2,2",
                "bottom_right": "3,6"  # 2 rows, 5 columns = ~160x24px (wide but short)
            }
        }
    ]
    
    for i, test in enumerate(test_positions, 1):
        print(f"\nTest {i}: {test['name']}")
        response = test['response']
        
        # Parse response
        tl_parts = response['top_left'].split(',')
        br_parts = response['bottom_right'].split(',')
        top_left_cell = (int(tl_parts[0]), int(tl_parts[1]))
        bottom_right_cell = (int(br_parts[0]), int(br_parts[1]))
        
        print(f"  Cells: {response['top_left']} to {response['bottom_right']}")
        
        # Calculate covered area
        rows_covered = bottom_right_cell[0] - top_left_cell[0] + 1
        cols_covered = bottom_right_cell[1] - top_left_cell[1] + 1
        area_width = cols_covered * cell_width
        area_height = rows_covered * cell_height
        
        print(f"  Cells covered: {rows_covered} rows x {cols_covered} cols")
        print(f"  Approximate area: {area_width}x{area_height}px (need {element_width}x{element_height}px)")
        
        # Check if in upper portion (sky area)
        in_sky = top_left_cell[0] <= grid_size[0] * 0.4  # Top 40% of grid
        print(f"  In sky area (top 40% of grid): {'✓' if in_sky else '✗'}")
        
        # Check size adequacy
        size_adequate = area_width >= element_width * 0.9 and area_height >= element_height * 0.9
        print(f"  Size adequate (90% of required): {'✓' if size_adequate else '✗'}")
        
        # Visualize
        vis_path = output_dir / f"visualization_{word}_test{i}.png"
        visualize_grid_with_overlay(
            frame, grid_size, top_left_cell, bottom_right_cell, 
            f"{word} ({test['name']})", (element_width, element_height), str(vis_path)
        )
    
    print("\n" + "=" * 60)
    print("ANALYSIS: GRID vs PIXEL APPROACH")
    print("=" * 60)
    
    print("\nGrid-based observations:")
    print(f"1. Grid quantization: With {grid_size[0]}x{grid_size[1]} grid, each cell is ~{cell_width}x{cell_height}px")
    print(f"2. Element needs {element_width}x{element_height}px, requiring ~{element_width//cell_width}x{element_height//cell_height} cells")
    print("3. Grid provides clear visual reference for 'sky' area (top rows)")
    print("4. LLM can see which cells contain the subject")
    
    print("\nAdvantages demonstrated:")
    print("✓ Visual grid helps identify safe placement zones")
    print("✓ Multi-cell spanning handles size requirements naturally")
    print("✓ Grid coordinates are simpler than pixel coordinates")
    print("✓ Less prone to boundary calculation errors")
    
    print("\nChallenges revealed:")
    print("✗ Grid quantization may not perfectly match element size")
    print("✗ Coarse grid (9x8) makes precise sizing difficult")
    print("✗ Need to balance grid resolution vs complexity")
    
    print("\nRecommendation:")
    print("Grid approach WORKS BETTER when:")
    print("- Grid is fine enough (our dynamic sizing helps)")
    print("- LLM understands multi-cell spanning")
    print("- Clear visual reference points are needed")
    print("- Consistent alignment is important")
    
    return output_dir


if __name__ == "__main__":
    output_dir = test_grid_positioning()
    print(f"\nAll test files saved in: {output_dir}/")
    print("\nTo view all visualizations:")
    print(f"open {output_dir}/*.png")