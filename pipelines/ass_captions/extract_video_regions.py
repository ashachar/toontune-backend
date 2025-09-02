#!/usr/bin/env python3
"""
Extract semantic regions from video using frame analysis.
This will identify safe zones for text placement.
"""

import os
import sys
import json
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple

def extract_video_regions(video_path: str, output_dir: str = None) -> str:
    """
    Extract semantic regions from video for intelligent text placement.
    
    Returns:
        Path to regions metadata JSON
    """
    video_path = Path(video_path)
    if not video_path.exists():
        print(f"Error: Video not found: {video_path}")
        return None
    
    # Set output directory
    if output_dir is None:
        output_dir = video_path.parent
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Output file
    video_name = video_path.stem
    regions_meta = output_dir / f"{video_name}_regions.json"
    
    # Check if already exists
    if regions_meta.exists():
        print(f"✅ Regions already analyzed: {regions_meta}")
        with open(regions_meta, 'r') as f:
            data = json.load(f)
        print(f"   Found {len(data['safe_zones'])} safe zones for text")
        return regions_meta
    
    print(f"\n🔍 Analyzing video regions: {video_path}")
    
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print("Error: Cannot open video")
        return None
    
    # Get properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    
    print(f"   Video: {width}x{height}, {duration:.1f}s")
    
    # Sample frames throughout video
    sample_interval = max(1, total_frames // 30)  # Sample ~30 frames
    
    # Analyze regions
    regions_data = analyze_video_regions(cap, sample_interval, width, height)
    
    cap.release()
    
    # Find safe zones for text placement
    safe_zones = find_safe_text_zones(regions_data, width, height)
    
    # Save metadata
    metadata = {
        "video_width": width,
        "video_height": height,
        "fps": fps,
        "duration": duration,
        "total_frames": total_frames,
        "safe_zones": safe_zones,
        "motion_areas": regions_data["motion_areas"],
        "static_areas": regions_data["static_areas"],
        "edge_density_map": regions_data["edge_density_map"]
    }
    
    with open(regions_meta, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✅ Regions analyzed and saved: {regions_meta}")
    print(f"   Found {len(safe_zones)} safe zones for text placement")
    
    # Print safe zones summary
    print("\n📍 Safe text zones:")
    for i, zone in enumerate(safe_zones[:5]):  # Show first 5
        print(f"   Zone {i+1}: {zone['name']} at ({zone['x']}, {zone['y']}) - score: {zone['score']:.2f}")
    
    return regions_meta


def analyze_video_regions(cap, sample_interval: int, width: int, height: int) -> Dict:
    """
    Analyze video to find motion areas, static regions, and edge density.
    """
    print("   Analyzing motion and structure...")
    
    # Initialize accumulators
    motion_accumulator = np.zeros((height, width), dtype=np.float32)
    edge_accumulator = np.zeros((height, width), dtype=np.float32)
    prev_gray = None
    frames_analyzed = 0
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx % sample_interval == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Motion detection
            if prev_gray is not None:
                flow = cv2.calcOpticalFlowFarneback(
                    prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
                )
                magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
                motion_accumulator += magnitude
            
            # Edge detection
            edges = cv2.Canny(gray, 50, 150)
            edge_accumulator += edges / 255.0
            
            prev_gray = gray
            frames_analyzed += 1
            
            if frames_analyzed % 10 == 0:
                print(f"      Analyzed {frames_analyzed} frames...")
        
        frame_idx += 1
    
    # Normalize accumulators
    if frames_analyzed > 0:
        motion_accumulator /= frames_analyzed
        edge_accumulator /= frames_analyzed
    
    # Find motion and static areas
    motion_threshold = np.percentile(motion_accumulator, 75)
    static_threshold = np.percentile(motion_accumulator, 25)
    
    motion_areas = []
    static_areas = []
    
    # Divide into grid cells
    grid_size = 8
    cell_h = height // grid_size
    cell_w = width // grid_size
    
    for i in range(grid_size):
        for j in range(grid_size):
            y1, y2 = i * cell_h, (i + 1) * cell_h
            x1, x2 = j * cell_w, (j + 1) * cell_w
            
            cell_motion = np.mean(motion_accumulator[y1:y2, x1:x2])
            cell_edges = np.mean(edge_accumulator[y1:y2, x1:x2])
            
            cell_info = {
                "x": (x1 + x2) // 2,
                "y": (y1 + y2) // 2,
                "width": cell_w,
                "height": cell_h,
                "motion_score": float(cell_motion),
                "edge_score": float(cell_edges)
            }
            
            if cell_motion > motion_threshold:
                motion_areas.append(cell_info)
            elif cell_motion < static_threshold:
                static_areas.append(cell_info)
    
    # Create edge density map (downsampled)
    edge_map_small = cv2.resize(edge_accumulator, (grid_size, grid_size))
    
    return {
        "motion_areas": motion_areas,
        "static_areas": static_areas,
        "edge_density_map": edge_map_small.tolist(),
        "frames_analyzed": frames_analyzed
    }


def find_safe_text_zones(regions_data: Dict, width: int, height: int) -> List[Dict]:
    """
    Find optimal zones for text placement based on motion and edge analysis.
    """
    safe_zones = []
    
    # Define candidate positions
    candidates = [
        # Traditional positions
        {"name": "top-center", "x": width // 2, "y": height // 6, "priority": 1.0},
        {"name": "bottom-center", "x": width // 2, "y": 5 * height // 6, "priority": 0.9},
        {"name": "middle-center", "x": width // 2, "y": height // 2, "priority": 0.7},
        
        # Offset positions
        {"name": "top-left", "x": width // 4, "y": height // 6, "priority": 0.8},
        {"name": "top-right", "x": 3 * width // 4, "y": height // 6, "priority": 0.8},
        {"name": "bottom-left", "x": width // 4, "y": 5 * height // 6, "priority": 0.7},
        {"name": "bottom-right", "x": 3 * width // 4, "y": 5 * height // 6, "priority": 0.7},
        
        # Upper/lower thirds
        {"name": "upper-third", "x": width // 2, "y": height // 3, "priority": 0.85},
        {"name": "lower-third", "x": width // 2, "y": 2 * height // 3, "priority": 0.85},
    ]
    
    # Score each candidate based on motion and edges
    for candidate in candidates:
        x, y = candidate["x"], candidate["y"]
        
        # Check if position is in a static area
        is_static = any(
            abs(area["x"] - x) < area["width"]/2 and 
            abs(area["y"] - y) < area["height"]/2
            for area in regions_data["static_areas"]
        )
        
        # Check if position is in a motion area
        is_motion = any(
            abs(area["x"] - x) < area["width"]/2 and 
            abs(area["y"] - y) < area["height"]/2
            for area in regions_data["motion_areas"]
        )
        
        # Calculate score
        score = candidate["priority"]
        
        if is_static:
            score += 0.3  # Bonus for static areas
        if is_motion:
            score -= 0.4  # Penalty for motion areas
        
        # Add position-based adjustments
        if "bottom" in candidate["name"]:
            score += 0.1  # Slight preference for bottom (less intrusive)
        if "center" in candidate["name"]:
            score += 0.05  # Slight preference for centered text
        
        safe_zones.append({
            "name": candidate["name"],
            "x": x,
            "y": y,
            "score": score,
            "is_static": is_static,
            "is_motion": is_motion,
            "suitable_for": classify_zone_usage(candidate["name"], score)
        })
    
    # Sort by score
    safe_zones.sort(key=lambda z: z["score"], reverse=True)
    
    return safe_zones


def classify_zone_usage(zone_name: str, score: float) -> List[str]:
    """
    Classify what type of content is suitable for each zone.
    """
    suitable = []
    
    if "top" in zone_name:
        suitable.append("title")
        suitable.append("important")
    if "bottom" in zone_name:
        suitable.append("subtitle")
        suitable.append("normal")
    if "center" in zone_name and score > 0.8:
        suitable.append("emphasis")
    if "third" in zone_name:
        suitable.append("subtitle")
        suitable.append("normal")
    
    return suitable


if __name__ == "__main__":
    # Test with ai_math1 video
    video_path = "../../uploads/assets/videos/ai_math1.mp4"
    
    regions_path = extract_video_regions(video_path)
    
    if regions_path:
        print("\n✅ Region analysis complete!")
        print("   This data will be used for intelligent text positioning")
