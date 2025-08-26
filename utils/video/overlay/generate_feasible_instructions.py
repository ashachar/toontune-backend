#!/usr/bin/env python3
"""
Generate feasible placement instructions based on actual available space in frames.
This preprocesses the video to determine where overlays CAN actually fit.
"""

import cv2
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple
import sys

sys.path.append(str(Path(__file__).parent.parent))
from get_overlay_pixels_by_grid import extract_frame_at_timestamp


def analyze_available_space(
    frame: np.ndarray,
    element_width: int,
    element_height: int
) -> List[str]:
    """
    Analyze frame to find where an overlay of given size can actually fit.
    Returns list of feasible placement descriptions.
    """
    h, w = frame.shape[:2]
    
    # For this specific video (256x114), woman occupies x:60-180, y:20-114
    # This is a simplified analysis - in production, would use actual segmentation
    
    feasible_placements = []
    
    # Check corners and edges
    regions = {
        "far left edge": (0, 0, 60, h),  # Left of subject
        "far right edge": (180, 0, w, h),  # Right of subject  
        "top left corner": (0, 0, 60, 60),
        "top right corner": (180, 0, w, 60),
        "above head area": (60, 0, 180, 20),  # Very thin strip above
    }
    
    for placement_name, (x1, y1, x2, y2) in regions.items():
        region_width = x2 - x1
        region_height = y2 - y1
        
        if region_width >= element_width and region_height >= element_height:
            feasible_placements.append(placement_name)
    
    # If no regions work, suggest alternatives
    if not feasible_placements:
        if element_width <= 60:  # Can fit in narrow edges
            feasible_placements.append("far left edge of frame")
            feasible_placements.append("far right edge of frame")
        else:
            feasible_placements.append("consider smaller overlay size or semi-transparent overlay")
    
    return feasible_placements


def update_instructions_for_feasibility(json_path: str, video_path: str) -> Dict:
    """
    Update the JSON instructions with feasible placement instructions.
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    updated_data = data.copy()
    
    for scene in updated_data['scenes']:
        if 'scene_description' not in scene:
            continue
            
        scene_desc = scene['scene_description']
        if 'text_overlays' not in scene_desc:
            continue
            
        for overlay in scene_desc['text_overlays']:
            # Extract frame at this timestamp
            timestamp = float(overlay['start_seconds'])
            frame = extract_frame_at_timestamp(video_path, timestamp)
            
            # Get element size
            element_width = overlay.get('size_pixels', {}).get('width', 100)
            element_height = overlay.get('size_pixels', {}).get('height', 50)
            
            # Find feasible placements
            feasible = analyze_available_space(frame, element_width, element_height)
            
            # Store original and update with feasible
            overlay['original_placement'] = overlay['placement']
            
            # Map original intent to feasible placement
            original = overlay['placement'].lower()
            
            if 'sky' in original or 'above' in original:
                # Sky/above placements - find nearest feasible alternative
                if "top right corner" in feasible:
                    overlay['placement'] = "top right corner away from subject"
                elif "top left corner" in feasible:
                    overlay['placement'] = "top left corner away from subject"
                elif "far right edge" in feasible:
                    overlay['placement'] = "far right edge of frame"
                else:
                    overlay['placement'] = feasible[0] if feasible else "semi-transparent overlay needed"
                    
            elif 'left' in original:
                if "far left edge" in feasible or "top left corner" in feasible:
                    overlay['placement'] = feasible[0]
                else:
                    overlay['placement'] = "far right edge as alternative"
                    
            elif 'right' in original:
                if "far right edge" in feasible or "top right corner" in feasible:
                    overlay['placement'] = feasible[0]
                else:
                    overlay['placement'] = "far left edge as alternative"
                    
            elif 'corner' in original:
                # Keep corner placements if feasible
                for f in feasible:
                    if 'corner' in f:
                        overlay['placement'] = f
                        break
                else:
                    overlay['placement'] = feasible[0] if feasible else "nearest edge"
                    
            else:
                # For complex descriptions, pick the best available
                overlay['placement'] = feasible[0] if feasible else "best available background area"
            
            overlay['feasible_placements'] = feasible
            
            print(f"Updated '{overlay['word']}' at {timestamp:.2f}s:")
            print(f"  Original: {overlay['original_placement']}")
            print(f"  Feasible: {overlay['placement']}")
            print(f"  Options: {feasible}")
            print()
    
    return updated_data


def main():
    """Update instructions and regenerate everything with feasible placements."""
    
    video_path = "uploads/assets/videos/do_re_mi_with_music/scenes/scene_001.mp4"
    original_json = "utils/video_overlay/do_re_mi_instructions_v2.json"
    
    print("=" * 60)
    print("GENERATING FEASIBLE PLACEMENT INSTRUCTIONS")
    print("=" * 60)
    print()
    
    # Update instructions
    updated_data = update_instructions_for_feasibility(original_json, video_path)
    
    # Save updated JSON
    output_path = "utils/video_overlay/do_re_mi_instructions_feasible.json"
    with open(output_path, 'w') as f:
        json.dump(updated_data, f, indent=2)
    
    print(f"\nUpdated instructions saved to: {output_path}")
    print("\nNow you can run:")
    print(f"python utils/video_overlay/get_overlay_pixels_by_grid.py {video_path} {output_path} --output-dir utils/video_overlay/grid_overlays_feasible --dry-run")
    print("\nThis will generate prompts with realistic, achievable placement instructions.")
    
    return output_path


if __name__ == "__main__":
    output_path = main()