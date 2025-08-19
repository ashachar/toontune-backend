#!/usr/bin/env python3
"""
Video Text Animator with Intelligent Placement

This script segments a video, analyzes the segments, and intelligently places
animated text based on segment locations and their descriptions.

Key features:
- Automatic video segmentation using SAM2
- Intelligent text placement based on segment analysis
- Dynamic animation selection based on context
- JSON-based animation instructions for rendering

Usage:
    python video_text_animator.py <video_path> <text_file> [output_dir]
    
Text file format:
    0.5: Hello world
    2.0: Welcome to the show
    5.5: Thanks for watching
"""

import time
import cv2
import numpy as np
from pathlib import Path
import replicate
import json
import requests
from PIL import Image
from io import BytesIO
import os
from dotenv import load_dotenv
import subprocess
import google.generativeai as genai
import sys
import re
from typing import List, Dict, Tuple, Any

# Load environment variables
load_dotenv()

# Import the existing segmentation functions
sys.path.append(str(Path(__file__).parent))
from video_segmentation.video_segmentation_and_annotation import (
    setup_gemini,
    check_replicate,
    segment_first_frame,
    create_two_asset_image,
    get_segment_descriptions
)

def parse_text_file(text_file_path: Path) -> List[Dict[str, Any]]:
    """
    Parse text file with timestamps and text to display
    Format: timestamp: text
    """
    texts = []
    with open(text_file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            match = re.match(r'^([\d.]+):\s*(.+)$', line)
            if match:
                timestamp = float(match.group(1))
                text = match.group(2).strip()
                texts.append({
                    'timestamp': timestamp,
                    'text': text
                })
    
    return sorted(texts, key=lambda x: x['timestamp'])

def get_available_animations() -> List[str]:
    """
    Get list of available animation functions from the animations directory
    """
    animations_dir = Path(__file__).parent / 'animations'
    animation_files = list(animations_dir.glob('*.py'))
    
    # Extract animation names (excluding __init__ and animate base class)
    animations = []
    for file in animation_files:
        name = file.stem
        if name not in ['__init__', 'animate', '__pycache__']:
            animations.append(name)
    
    return animations

def analyze_segments_for_placement(segments: List[Dict], frame_shape: Tuple[int, int]) -> Dict:
    """
    Analyze segments to determine safe zones for text placement
    """
    height, width = frame_shape[:2]
    
    # Create occupancy grid (divide frame into regions)
    grid_rows, grid_cols = 6, 8
    cell_height = height // grid_rows
    cell_width = width // grid_cols
    
    occupancy_grid = np.zeros((grid_rows, grid_cols), dtype=float)
    
    # Mark occupied cells based on segments
    for seg in segments:
        mask = seg['mask']
        for row in range(grid_rows):
            for col in range(grid_cols):
                y1 = row * cell_height
                y2 = min((row + 1) * cell_height, height)
                x1 = col * cell_width
                x2 = min((col + 1) * cell_width, width)
                
                cell_mask = mask[y1:y2, x1:x2]
                occupancy_ratio = np.sum(cell_mask) / (cell_mask.size + 1)
                occupancy_grid[row, col] = occupancy_ratio
    
    # Find safe zones (areas with low occupancy)
    safe_zones = []
    for row in range(grid_rows):
        for col in range(grid_cols):
            if occupancy_grid[row, col] < 0.2:  # Less than 20% occupied
                center_x = col * cell_width + cell_width // 2
                center_y = row * cell_height + cell_height // 2
                region_y_idx = min(row // max(1, grid_rows // 3), 2)
                region_x_idx = min(col // max(1, grid_cols // 3), 2)
                safe_zones.append({
                    'x': center_x,
                    'y': center_y,
                    'score': 1.0 - occupancy_grid[row, col],
                    'region': f"{['top', 'middle', 'bottom'][region_y_idx]}-{['left', 'center', 'right'][region_x_idx]}"
                })
    
    # Categorize segments by location and importance
    segment_info = {
        'segments': [],
        'safe_zones': sorted(safe_zones, key=lambda x: x['score'], reverse=True),
        'occupancy_grid': occupancy_grid.tolist(),
        'frame_dimensions': {'width': width, 'height': height}
    }
    
    for seg in segments:
        cx, cy = seg['centroid']
        region_y = 'top' if cy < height/3 else 'middle' if cy < 2*height/3 else 'bottom'
        region_x = 'left' if cx < width/3 else 'center' if cx < 2*width/3 else 'right'
        
        segment_info['segments'].append({
            'id': seg['id'],
            'description': seg.get('description', ''),
            'centroid': {'x': int(cx), 'y': int(cy)},
            'area': int(seg['area']),
            'region': f"{region_y}-{region_x}",
            'relative_size': float(seg['area'] / (width * height))
        })
    
    return segment_info

def generate_animation_strategy(segment_info: Dict, texts: List[Dict], model, animations: List[str]) -> str:
    """
    Use Gemini to generate animation strategy based on segment analysis
    """
    print(f"\n[{time.strftime('%H:%M:%S')}] Generating animation strategy with Gemini...")
    
    prompt = f"""You are an expert video text animator. Given a video's segment analysis and text to display,
determine the optimal animation strategy for each text element.

VIDEO ANALYSIS:
- Frame dimensions: {segment_info['frame_dimensions']['width']}x{segment_info['frame_dimensions']['height']}
- Number of segments: {len(segment_info['segments'])}
- Main segments:
{json.dumps(segment_info['segments'][:5], indent=2)}

- Safe zones for text placement (top 5):
{json.dumps(segment_info['safe_zones'][:5], indent=2)}

TEXTS TO ANIMATE (with EXACT timestamps when they should appear):
{json.dumps(texts, indent=2)}

AVAILABLE ANIMATIONS:
Entry animations: {', '.join([a for a in animations if 'in' in a.lower() or 'emergence' in a.lower()])}
Exit animations: {', '.join([a for a in animations if 'out' in a.lower() or 'submerge' in a.lower()])}

CRITICAL RULES:
1. Use the EXACT timestamp from the input as start_time (DO NOT CHANGE IT)
2. Duration should be 0.4-0.6 seconds for quick appearance
3. Calculate duration to ensure text disappears before next text (no overlap)
4. Place each text in a DIFFERENT safe zone to add variety
5. Avoid placing text at the very edges - keep at least 50px margin
6. For water/ocean scenes, prefer 'emergence_from_static_point' and 'submerge_to_static_point'
7. Vary the Y position - don't put all text at the same height
8. Last text should end before video ends

Provide a JSON response with animation instructions for each text:
{{
    "animations": [
        {{
            "text": "the actual text",
            "start_time": <MUST match input timestamp exactly>,
            "duration": <1.5-2.5 seconds, ensure no overlap>,
            "position": {{"x": <vary across frame>, "y": <vary heights, not all at top>}},
            "entry_animation": "<animation_name>",
            "exit_animation": "<animation_name>",
            "font_size": <int 30-50>,
            "color": [R, G, B],
            "reasoning": "brief explanation"
        }}
    ]
}}

Ensure all animation names exactly match the available animations list.
Position coordinates should be within frame dimensions with margins.
Colors should provide good contrast with the background."""
    
    try:
        response = model.generate_content(prompt)
        output = response.text
        
        # Extract JSON from response
        if "```json" in output:
            output = output.split("```json")[1].split("```")[0]
        elif "```" in output:
            output = output.split("```")[1].split("```")[0]
        
        strategy = json.loads(output.strip())
        return strategy
        
    except Exception as e:
        print(f"Error generating strategy: {e}")
        # Fallback strategy
        fallback = {"animations": []}
        for i, text_item in enumerate(texts):
            fallback["animations"].append({
                "text": text_item['text'],
                "start_time": text_item['timestamp'],
                "duration": 3.0,
                "position": {"x": segment_info['frame_dimensions']['width'] // 2,
                           "y": segment_info['frame_dimensions']['height'] - 100},
                "entry_animation": "fade_in",
                "exit_animation": "fade_out",
                "font_size": 40,
                "color": [255, 255, 255],
                "reasoning": "Fallback placement"
            })
        return fallback

def render_animated_video(video_path: Path, animation_strategy: Dict, output_path: Path):
    """
    Render the final video with animated text overlays
    """
    print(f"\n[{time.strftime('%H:%M:%S')}] Rendering animated video...")
    
    cap = cv2.VideoCapture(str(video_path))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    animations = animation_strategy['animations']
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        current_time = frame_idx / fps
        
        # Apply text animations for current frame
        for anim in animations:
            start_time = anim['start_time']
            end_time = start_time + anim['duration']
            
            if start_time <= current_time <= end_time:
                # Calculate animation progress
                progress = (current_time - start_time) / anim['duration']
                
                # Apply entry animation (first 20% of duration)
                if progress < 0.2:
                    alpha = progress / 0.2
                elif progress > 0.8:  # Exit animation (last 20% of duration)
                    alpha = (1.0 - progress) / 0.2
                else:
                    alpha = 1.0
                
                # Draw text with alpha blending
                text = anim['text']
                pos = (int(anim['position']['x']), int(anim['position']['y']))
                font_size = anim['font_size'] / 30  # OpenCV font scale
                color = tuple(anim['color'])
                
                # Create overlay for text
                overlay = frame.copy()
                
                # Add background rectangle for better readability
                (text_width, text_height), _ = cv2.getTextSize(
                    text, cv2.FONT_HERSHEY_SIMPLEX, font_size, 2
                )
                padding = 10
                cv2.rectangle(overlay,
                            (pos[0] - text_width//2 - padding, pos[1] - text_height - padding),
                            (pos[0] + text_width//2 + padding, pos[1] + padding),
                            (0, 0, 0), -1)
                
                # Draw text
                cv2.putText(overlay, text,
                          (pos[0] - text_width//2, pos[1]),
                          cv2.FONT_HERSHEY_SIMPLEX, font_size,
                          color, 2, cv2.LINE_AA)
                
                # Blend with original frame
                frame = cv2.addWeighted(frame, 1 - alpha * 0.7, overlay, alpha * 0.7, 0)
        
        out.write(frame)
        frame_idx += 1
        
        if frame_idx % 30 == 0:
            print(f"[{time.strftime('%H:%M:%S')}] Processed {frame_idx}/{total_frames} frames")
    
    cap.release()
    out.release()
    
    # Convert to H.264
    h264_path = output_path.parent / f"{output_path.stem}_h264.mp4"
    subprocess.run([
        'ffmpeg', '-i', str(output_path),
        '-c:v', 'libx264', '-preset', 'medium', '-crf', '23',
        '-c:a', 'copy', '-y', str(h264_path)
    ], capture_output=True)
    
    return h264_path

def save_animation_strategy(strategy: Dict, output_dir: Path):
    """
    Save the animation strategy to a JSON file
    """
    strategy_path = output_dir / "animation_strategy.json"
    with open(strategy_path, 'w') as f:
        json.dump(strategy, f, indent=2)
    print(f"[{time.strftime('%H:%M:%S')}] Animation strategy saved to {strategy_path}")
    return strategy_path

def main():
    print("[START] Video Text Animator Pipeline")
    
    # Setup
    model = setup_gemini()
    check_replicate()
    
    # Parse arguments
    if len(sys.argv) < 3:
        print("Usage: python video_text_animator.py <video_path> <text_file> [output_dir]")
        sys.exit(1)
    
    video_path = Path(sys.argv[1])
    text_file = Path(sys.argv[2])
    
    if not video_path.exists():
        print(f"Error: Video file not found: {video_path}")
        sys.exit(1)
    
    if not text_file.exists():
        print(f"Error: Text file not found: {text_file}")
        sys.exit(1)
    
    output_dir = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("output/video_text_animator")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Parse text file
    print(f"\n[{time.strftime('%H:%M:%S')}] Parsing text file...")
    texts = parse_text_file(text_file)
    print(f"Found {len(texts)} text entries")
    
    # Get available animations
    animations = get_available_animations()
    print(f"Available animations: {', '.join(animations)}")
    
    # Extract first frame for analysis
    print(f"\n[{time.strftime('%H:%M:%S')}] Extracting first frame...")
    cap = cv2.VideoCapture(str(video_path))
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("Failed to read video")
        sys.exit(1)
    
    frame_path = output_dir / "frame0.png"
    cv2.imwrite(str(frame_path), frame)
    
    # Segment the first frame
    segments = segment_first_frame(frame_path)
    if not segments:
        print("Segmentation failed")
        sys.exit(1)
    
    # Create two-asset image for Gemini
    concat_path = create_two_asset_image(frame, segments, output_dir)
    
    # Get segment descriptions
    segments = get_segment_descriptions(concat_path, segments, model)
    
    # Analyze segments for text placement
    segment_info = analyze_segments_for_placement(segments, frame.shape)
    
    # Generate animation strategy
    animation_strategy = generate_animation_strategy(segment_info, texts, model, animations)
    
    # Save animation strategy
    strategy_path = save_animation_strategy(animation_strategy, output_dir)
    
    # Render the animated video
    temp_output = output_dir / "animated_video.mp4"
    final_video = render_animated_video(video_path, animation_strategy, temp_output)
    
    # Print final report
    print(f"\n{'='*60}")
    print(f"✅ VIDEO TEXT ANIMATION COMPLETE")
    print(f"{'='*60}")
    print(f"\nSegments analyzed: {len(segments)}")
    print(f"Text entries animated: {len(texts)}")
    print(f"\nOutput files:")
    print(f"  - Animation strategy: {strategy_path}")
    print(f"  - Final video: {final_video}")
    print(f"  - Segment analysis: {output_dir}/concatenated_input.png")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()