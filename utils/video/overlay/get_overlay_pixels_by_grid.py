#!/usr/bin/env python3
"""
Script to determine optimal overlay placement using dynamic grid-based positioning.
Extracts frames from video, adds grid overlay without text, and uses AI to determine placement.
Grid size is calculated based on element dimensions.
"""

import cv2
import numpy as np
import argparse
import json
import yaml
from pathlib import Path
import tempfile
import os
from typing import Tuple, Dict, List, Optional
import random
import math

# Add parent directory to path for imports
import sys
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


def calculate_grid_size(
    element_width: int,
    element_height: int,
    scene_width: int = 1920,
    scene_height: int = 1080
) -> Tuple[int, int]:
    """
    Calculate optimal grid size based on element dimensions.
    
    Grid cells should be:
    - At least 2x the size of the element
    - At least 1/10 of the scene dimensions
    
    Returns:
        (rows, cols) for the grid
    """
    # Calculate minimum cell size
    min_cell_width = max(element_width * 2, scene_width // 10)
    min_cell_height = max(element_height * 2, scene_height // 10)
    
    # Calculate grid dimensions
    cols = max(3, scene_width // min_cell_width)
    rows = max(3, scene_height // min_cell_height)
    
    # Limit grid to reasonable size
    cols = min(20, cols)
    rows = min(15, rows)
    
    return (rows, cols)


def create_grid_overlay(
    frame: np.ndarray, 
    grid_size: Tuple[int, int],
    line_color: Tuple[int, int, int] = (0, 255, 0),
    line_thickness: int = 1
) -> Tuple[np.ndarray, Dict[Tuple[int, int], Tuple[int, int]]]:
    """
    Add a grid overlay to the frame WITHOUT text labels.
    
    Args:
        frame: Input frame
        grid_size: (rows, cols) for the grid
        line_color: Color of grid lines (BGR)
        line_thickness: Thickness of grid lines
        
    Returns:
        Frame with grid overlay and dictionary mapping (row, col) to center pixels
    """
    h, w = frame.shape[:2]
    rows, cols = grid_size
    
    # Calculate cell dimensions
    cell_height = h // rows
    cell_width = w // cols
    
    # Create output frame
    output = frame.copy()
    
    # Dictionary to store cell centers
    cell_centers = {}
    
    # Draw grid lines
    # Vertical lines
    for col in range(1, cols):
        x = col * cell_width
        cv2.line(output, (x, 0), (x, h), line_color, line_thickness)
    
    # Horizontal lines
    for row in range(1, rows):
        y = row * cell_height
        cv2.line(output, (0, y), (w, y), line_color, line_thickness)
    
    # Draw border
    cv2.rectangle(output, (0, 0), (w-1, h-1), line_color, line_thickness)
    
    # Calculate cell centers (using 1-based indexing for user-friendly coordinates)
    for row in range(rows):
        for col in range(cols):
            # Calculate cell boundaries
            x1 = col * cell_width
            y1 = row * cell_height
            x2 = min((col + 1) * cell_width, w)
            y2 = min((row + 1) * cell_height, h)
            
            # Calculate center point
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            
            # Store with 1-based indexing (row, col)
            cell_centers[(row + 1, col + 1)] = (center_x, center_y)
    
    return output, cell_centers


def get_pixel_from_cell(
    cell_centers: Dict[Tuple[int, int], Tuple[int, int]], 
    cell_row: int,
    cell_col: int,
    variation: int = 10
) -> Tuple[int, int]:
    """
    Get a pixel position from cell coordinates with slight randomization.
    
    Args:
        cell_centers: Dictionary mapping (row, col) to center pixels
        cell_row: Row number (1-based)
        cell_col: Column number (1-based)
        variation: Random variation in pixels
        
    Returns:
        Selected pixel position (x, y)
    """
    if (cell_row, cell_col) not in cell_centers:
        # If cell doesn't exist, return center of available cells
        all_centers = list(cell_centers.values())
        center_x = sum(x for x, y in all_centers) // len(all_centers)
        center_y = sum(y for x, y in all_centers) // len(all_centers)
        return (center_x, center_y)
    
    center_x, center_y = cell_centers[(cell_row, cell_col)]
    
    # Add some randomness within the cell
    final_x = center_x + random.randint(-variation, variation)
    final_y = center_y + random.randint(-variation, variation)
    
    return (final_x, final_y)


def add_prompts_to_yaml(prompts_data: Dict):
    """Add overlay positioning prompts to prompts.yaml file."""
    prompts_path = Path(__file__).parent.parent.parent / "prompts.yaml"
    
    # Load existing prompts
    if prompts_path.exists():
        with open(prompts_path, 'r') as f:
            existing_data = yaml.safe_load(f) or {}
    else:
        existing_data = {}
    
    # Add or update the overlay positioning section
    if 'prompts' not in existing_data:
        existing_data['prompts'] = {}
    
    existing_data['prompts']['overlay_positioning'] = prompts_data
    
    # Save back to file
    with open(prompts_path, 'w') as f:
        yaml.dump(existing_data, f, default_flow_style=False, sort_keys=False)
    
    return prompts_path


def process_video_overlays(
    video_path: str,
    scenes_data: List[Dict],
    output_dir: str,
    dry_run: bool = True
) -> Dict:
    """
    Process all text overlays in the video and generate grid frames.
    
    Args:
        video_path: Path to the input video
        scenes_data: List of scene dictionaries with text overlay information
        output_dir: Directory to save grid frames
        dry_run: If True, don't call AI API, just generate grids
        
    Returns:
        Dictionary with overlay positioning information
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories for prompts and inferences
    prompts_dir = output_dir / 'prompts'
    inferences_dir = output_dir / 'inferences'
    prompts_dir.mkdir(exist_ok=True)
    inferences_dir.mkdir(exist_ok=True)
    
    # Downsample video for processing
    print(f"Downsampling video for grid processing...")
    temp_video = tempfile.mktemp(suffix='.mp4')
    downsampler = VideoDownsampler(video_path, temp_video, preset='small')
    downsampler.downsample()
    
    results = {
        'video_path': video_path,
        'overlays': []
    }
    
    # Prepare prompts for YAML
    prompts_data = {
        'description': 'Grid-based overlay positioning for video text elements',
        'model': 'gemini-2.0-flash-exp',
        'purpose': 'Determine optimal grid cells for text overlay placement',
        'base_prompt': """Given a video frame with a numbered grid overlay, determine the best grid cells for placing the text overlay.

The overlay may span multiple grid cells. Provide the top-left and bottom-right corner cells.

Grid coordinates are in format (row, column) where:
- Row 1 is at the top, increasing downward
- Column 1 is at the left, increasing rightward
- Example: (3,5) means 3rd row from top, 5th column from left

Placement instruction: {placement}
Text to place: "{word}"
Interaction style: {interaction_style}
Element size: {element_width}x{element_height} pixels
Grid size: {grid_rows}x{grid_cols} cells

Analyze the frame and determine:
1. Top-left corner cell for the overlay
2. Bottom-right corner cell for the overlay

If the overlay fits within a single cell, both corners should be the same cell.
If the overlay needs more space, specify different cells to create a bounding rectangle.

Return ONLY a JSON object in this exact format:
{{
  "top_left": "row,column",
  "bottom_right": "row,column"
}}

Example responses:
- Single cell: {{"top_left": "2,3", "bottom_right": "2,3"}}
- Multiple cells: {{"top_left": "2,3", "bottom_right": "4,5"}}
- Wide text: {{"top_left": "5,2", "bottom_right": "5,8"}}

Consider:
- The placement description provided
- Visual balance and composition
- Text/element size relative to grid cells
- Avoiding occlusion of important subjects
- The interaction style (whether it should move with characters or stay anchored)""",
        'instances': []
    }
    
    overlay_count = 0
    
    for scene_idx, scene in enumerate(scenes_data):
        if 'scene_description' not in scene:
            continue
            
        scene_desc = scene['scene_description']
        
        # Get scene dimensions from effects or use defaults
        scene_width = 1920
        scene_height = 1080
        if 'suggested_effects' in scene_desc and scene_desc['suggested_effects']:
            effect = scene_desc['suggested_effects'][0]
            if 'size_pixels' in effect:
                scene_width = effect['size_pixels']['width']
                scene_height = effect['size_pixels']['height']
        
        if 'text_overlays' not in scene_desc:
            continue
        
        for overlay in scene_desc['text_overlays']:
            overlay_count += 1
            
            # Extract frame at overlay start time
            timestamp = float(overlay['start_seconds'])
            frame = extract_frame_at_timestamp(temp_video, timestamp)
            
            # Get element dimensions
            element_width = overlay.get('size_pixels', {}).get('width', 100)
            element_height = overlay.get('size_pixels', {}).get('height', 50)
            
            # Calculate grid size based on element
            grid_size = calculate_grid_size(
                element_width, 
                element_height,
                scene_width,
                scene_height
            )
            
            # Create grid overlay
            grid_frame, cell_centers = create_grid_overlay(frame, grid_size)
            
            # Save grid frame
            grid_filename = f"grid_scene{scene_idx+1}_overlay{overlay_count}_{overlay['word']}.png"
            grid_path = output_dir / grid_filename
            cv2.imwrite(str(grid_path), grid_frame)
            
            # Create prompt instance
            prompt_instance = {
                'overlay_id': f"scene{scene_idx+1}_overlay{overlay_count}_{overlay['word']}",
                'timestamp': timestamp,
                'word': overlay['word'],
                'placement': overlay['placement'],
                'interaction_style': overlay.get('interaction_style', 'anchored_to_background'),
                'grid_size': f"{grid_size[0]}x{grid_size[1]}",
                'element_size': f"{element_width}x{element_height}"
            }
            prompts_data['instances'].append(prompt_instance)
            
            # Generate the actual prompt text
            prompt_text = prompts_data['base_prompt'].format(
                placement=overlay['placement'],
                word=overlay['word'],
                interaction_style=overlay.get('interaction_style', 'anchored_to_background'),
                element_width=element_width,
                element_height=element_height,
                grid_rows=grid_size[0],
                grid_cols=grid_size[1]
            )
            
            # Save prompt to file
            prompt_filename = f"prompt_scene{scene_idx+1}_overlay{overlay_count}_{overlay['word']}.txt"
            prompt_path = prompts_dir / prompt_filename
            with open(prompt_path, 'w') as f:
                f.write(prompt_text)
            
            print(f"  Prompt saved: {prompt_path}")
            
            overlay_info = {
                'scene_index': scene_idx,
                'word': overlay['word'],
                'timestamp': timestamp,
                'placement_description': overlay['placement'],
                'interaction_style': overlay.get('interaction_style', 'anchored_to_background'),
                'grid_frame': str(grid_path),
                'prompt_file': str(prompt_path),
                'grid_size': grid_size,
                'element_size': (element_width, element_height),
                'scene_size': (scene_width, scene_height)
            }
            
            if not dry_run:
                # TODO: Call Gemini API here with the prompt
                # For now, simulate with a response that might span multiple cells
                top_left_row = random.randint(1, max(1, grid_size[0] - 1))
                top_left_col = random.randint(1, max(1, grid_size[1] - 1))
                
                # Sometimes make it span multiple cells
                if random.random() > 0.5:  # 50% chance of spanning multiple cells
                    bottom_right_row = min(grid_size[0], top_left_row + random.randint(0, 2))
                    bottom_right_col = min(grid_size[1], top_left_col + random.randint(0, 3))
                else:
                    bottom_right_row = top_left_row
                    bottom_right_col = top_left_col
                
                # Calculate pixel positions for the bounding box
                top_left_pixel = get_pixel_from_cell(cell_centers, top_left_row, top_left_col, variation=0)
                bottom_right_pixel = get_pixel_from_cell(cell_centers, bottom_right_row, bottom_right_col, variation=0)
                
                overlay_info['cells'] = {
                    'top_left': (top_left_row, top_left_col),
                    'bottom_right': (bottom_right_row, bottom_right_col)
                }
                overlay_info['pixel_bounds'] = {
                    'top_left': top_left_pixel,
                    'bottom_right': bottom_right_pixel
                }
                
                # Save inference result
                inference_result = {
                    'overlay_id': f"scene{scene_idx+1}_overlay{overlay_count}_{overlay['word']}",
                    'timestamp': timestamp,
                    'response': {
                        'top_left': f"{top_left_row},{top_left_col}",
                        'bottom_right': f"{bottom_right_row},{bottom_right_col}"
                    },
                    'pixel_bounds': {
                        'top_left': top_left_pixel,
                        'bottom_right': bottom_right_pixel
                    },
                    'prompt_used': prompt_text
                }
                
                inference_filename = f"inference_scene{scene_idx+1}_overlay{overlay_count}_{overlay['word']}.json"
                inference_path = inferences_dir / inference_filename
                with open(inference_path, 'w') as f:
                    json.dump(inference_result, f, indent=2)
                
                overlay_info['inference_file'] = str(inference_path)
                print(f"  Inference saved: {inference_path}")
            else:
                overlay_info['cells'] = 'DRY_RUN'
                overlay_info['pixel_bounds'] = 'DRY_RUN'
                overlay_info['inference_file'] = 'DRY_RUN'
            
            results['overlays'].append(overlay_info)
            
            print(f"Generated grid for: Scene {scene_idx+1}, Overlay '{overlay['word']}' at {timestamp:.2f}s")
            print(f"  Grid size: {grid_size[0]}x{grid_size[1]} (element: {element_width}x{element_height}px)")
            print(f"  Grid frame: {grid_path}")
    
    # Clean up temp video
    if os.path.exists(temp_video):
        os.remove(temp_video)
    
    # Add prompts to YAML
    prompts_path = add_prompts_to_yaml(prompts_data)
    print(f"\nPrompts added to: {prompts_path}")
    
    # Save results
    results_path = output_dir / 'overlay_positions.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nTotal overlays processed: {overlay_count}")
    print(f"Results saved to: {results_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Generate grid overlays for text positioning')
    parser.add_argument('video_path', help='Path to the input video')
    parser.add_argument('scenes_json', help='Path to the scenes JSON file')
    parser.add_argument('--output-dir', default='grid_overlays_v2', help='Output directory for grids')
    parser.add_argument('--dry-run', action='store_true', help='Skip AI API calls')
    
    args = parser.parse_args()
    
    # Load scenes data
    with open(args.scenes_json, 'r') as f:
        scenes_data = json.load(f)
    
    # Process video overlays
    if 'scenes' in scenes_data:
        scenes = scenes_data['scenes']
    else:
        scenes = [scenes_data]  # Single scene
    
    results = process_video_overlays(
        args.video_path,
        scenes,
        args.output_dir,
        dry_run=args.dry_run
    )
    
    print("\nProcessing complete!")
    print(f"Grid frames saved to: {args.output_dir}/")
    if args.dry_run:
        print("DRY RUN MODE: No AI API calls were made.")
        print("Review the generated grid frames before running without --dry-run")


if __name__ == '__main__':
    main()