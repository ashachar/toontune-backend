#!/usr/bin/env python3
"""
Script to determine optimal overlay placement using grid-based positioning.
Extracts frames from video, adds grid overlay, and uses AI to determine placement.
"""

import cv2
import numpy as np
import argparse
import json
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


def create_grid_overlay(
    frame: np.ndarray, 
    grid_size: Tuple[int, int] = (10, 20),
    font_scale: float = 0.4,
    thickness: int = 1
) -> Tuple[np.ndarray, Dict[int, Tuple[int, int]]]:
    """
    Add a numbered grid overlay to the frame.
    
    Args:
        frame: Input frame
        grid_size: (rows, cols) for the grid
        font_scale: Font scale for numbers
        thickness: Line thickness
        
    Returns:
        Frame with grid overlay and dictionary mapping cell numbers to center pixels
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
    
    # Draw grid and add numbers
    cell_number = 1
    for row in range(rows):
        for col in range(cols):
            # Calculate cell boundaries
            x1 = col * cell_width
            y1 = row * cell_height
            x2 = min((col + 1) * cell_width, w)
            y2 = min((row + 1) * cell_height, h)
            
            # Draw cell borders
            cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 1)
            
            # Calculate center point
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            cell_centers[cell_number] = (center_x, center_y)
            
            # Add transparent background for number visibility
            text = str(cell_number)
            font = cv2.FONT_HERSHEY_SIMPLEX
            text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
            
            # Create semi-transparent background
            bg_x1 = center_x - text_size[0] // 2 - 2
            bg_y1 = center_y - text_size[1] // 2 - 2
            bg_x2 = center_x + text_size[0] // 2 + 2
            bg_y2 = center_y + text_size[1] // 2 + 2
            
            # Draw semi-transparent background
            overlay = output.copy()
            cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), -1)
            output = cv2.addWeighted(output, 0.7, overlay, 0.3, 0)
            
            # Draw cell number
            cv2.putText(
                output, text,
                (center_x - text_size[0] // 2, center_y + text_size[1] // 2),
                font, font_scale, (255, 255, 255), thickness
            )
            
            cell_number += 1
    
    return output, cell_centers


def get_pixel_with_decay(
    cell_centers: Dict[int, Tuple[int, int]], 
    target_cell: int, 
    decay_factor: float = 0.8
) -> Tuple[int, int]:
    """
    Get a pixel position with decaying probability based on distance from target cell.
    
    Args:
        cell_centers: Dictionary mapping cell numbers to center pixels
        target_cell: The target cell number
        decay_factor: How quickly probability decays with distance
        
    Returns:
        Selected pixel position (x, y)
    """
    if target_cell not in cell_centers:
        # If target cell doesn't exist, return center of available cells
        all_centers = list(cell_centers.values())
        center_x = sum(x for x, y in all_centers) // len(all_centers)
        center_y = sum(y for x, y in all_centers) // len(all_centers)
        return (center_x, center_y)
    
    target_x, target_y = cell_centers[target_cell]
    
    # Add some randomness within the cell
    cell_variation = 10  # pixels
    final_x = target_x + random.randint(-cell_variation, cell_variation)
    final_y = target_y + random.randint(-cell_variation, cell_variation)
    
    return (final_x, final_y)


def create_prompt_for_overlay(
    overlay_text: str,
    placement_description: str,
    interaction_style: str
) -> str:
    """Create a prompt for determining grid cell placement."""
    prompt = f"""Given a video frame with a numbered grid overlay, determine the best grid cell number for placing the text "{overlay_text}".

Placement instruction: {placement_description}
Interaction style: {interaction_style}

Analyze the frame and return ONLY the grid cell number (e.g., "42") where this text should be placed.
Consider:
- The placement description provided
- Visual balance and composition
- Avoiding occlusion of important subjects
- The interaction style (whether it should move with characters or stay anchored)

Return only the cell number, nothing else."""
    
    return prompt


def process_video_overlays(
    video_path: str,
    scenes_data: List[Dict],
    output_dir: str,
    grid_size: Tuple[int, int] = (10, 20),
    dry_run: bool = True
) -> Dict:
    """
    Process all text overlays in the video and generate grid frames.
    
    Args:
        video_path: Path to the input video
        scenes_data: List of scene dictionaries with text overlay information
        output_dir: Directory to save grid frames
        grid_size: Size of the grid (rows, cols)
        dry_run: If True, don't call AI API, just generate grids
        
    Returns:
        Dictionary with overlay positioning information
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Downsample video for processing
    print(f"Downsampling video for grid processing...")
    temp_video = tempfile.mktemp(suffix='.mp4')
    downsampler = VideoDownsampler(video_path, temp_video, preset='small')
    downsampler.downsample()
    
    results = {
        'video_path': video_path,
        'grid_size': grid_size,
        'overlays': []
    }
    
    overlay_count = 0
    
    for scene_idx, scene in enumerate(scenes_data):
        if 'scene_description' not in scene:
            continue
            
        scene_desc = scene['scene_description']
        if 'text_overlays' not in scene_desc:
            continue
        
        for overlay in scene_desc['text_overlays']:
            overlay_count += 1
            
            # Extract frame at overlay start time
            timestamp = float(overlay['start_seconds'])
            frame = extract_frame_at_timestamp(temp_video, timestamp)
            
            # Create grid overlay
            grid_frame, cell_centers = create_grid_overlay(frame, grid_size)
            
            # Save grid frame
            grid_filename = f"grid_scene{scene_idx+1}_overlay{overlay_count}_{overlay['word']}.png"
            grid_path = output_dir / grid_filename
            cv2.imwrite(str(grid_path), grid_frame)
            
            # Create prompt
            prompt = create_prompt_for_overlay(
                overlay['word'],
                overlay['placement'],
                overlay.get('interaction_style', 'anchored_to_background')
            )
            
            # Save prompt
            prompt_filename = f"prompt_scene{scene_idx+1}_overlay{overlay_count}_{overlay['word']}.txt"
            prompt_path = output_dir / prompt_filename
            with open(prompt_path, 'w') as f:
                f.write(prompt)
            
            overlay_info = {
                'scene_index': scene_idx,
                'word': overlay['word'],
                'timestamp': timestamp,
                'placement_description': overlay['placement'],
                'interaction_style': overlay.get('interaction_style', 'anchored_to_background'),
                'grid_frame': str(grid_path),
                'prompt_file': str(prompt_path),
                'prompt': prompt
            }
            
            if not dry_run:
                # TODO: Call Gemini API here
                # For now, simulate with a random cell
                target_cell = random.randint(1, grid_size[0] * grid_size[1])
                pixel_position = get_pixel_with_decay(cell_centers, target_cell)
                overlay_info['target_cell'] = target_cell
                overlay_info['pixel_position'] = pixel_position
            else:
                overlay_info['target_cell'] = 'DRY_RUN'
                overlay_info['pixel_position'] = 'DRY_RUN'
            
            results['overlays'].append(overlay_info)
            
            print(f"Generated grid for: Scene {scene_idx+1}, Overlay '{overlay['word']}' at {timestamp:.2f}s")
            print(f"  Grid frame: {grid_path}")
            print(f"  Prompt saved: {prompt_path}")
    
    # Clean up temp video
    if os.path.exists(temp_video):
        os.remove(temp_video)
    
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
    parser.add_argument('--output-dir', default='grid_overlays', help='Output directory for grids')
    parser.add_argument('--grid-rows', type=int, default=10, help='Number of grid rows')
    parser.add_argument('--grid-cols', type=int, default=20, help='Number of grid columns')
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
        grid_size=(args.grid_rows, args.grid_cols),
        dry_run=args.dry_run
    )
    
    print("\nProcessing complete!")
    if args.dry_run:
        print("DRY RUN MODE: No AI API calls were made.")
        print("Review the generated grid frames and prompts before running without --dry-run")


if __name__ == '__main__':
    main()