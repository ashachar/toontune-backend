#!/usr/bin/env python3
"""
Extract SAM2 video segmentation masks for the entire video.
Saves segment masks and metadata to the video's folder.
"""

import os
import sys
import json
import cv2
import numpy as np
import replicate
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def extract_sam2_video_segments(video_path, output_dir=None):
    """
    Extract SAM2 automatic video segmentation for semantic understanding.
    
    Args:
        video_path: Path to input video
        output_dir: Where to save masks (defaults to video's directory)
    
    Returns:
        Path to segmentation metadata JSON
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
    
    # Output files
    video_name = video_path.stem
    segments_video = output_dir / f"{video_name}_sam2_segments.mp4"
    segments_meta = output_dir / f"{video_name}_sam2_segments.json"
    
    # Check if already exists
    if segments_video.exists() and segments_meta.exists():
        print(f"✅ SAM2 segments already exist:")
        print(f"   Video: {segments_video}")
        print(f"   Metadata: {segments_meta}")
        return segments_meta
    
    print(f"\n🎯 Extracting SAM2 video segments for: {video_path}")
    print("This will identify semantic regions throughout the video...")
    
    try:
        # For now, we'll use a grid-based approach to get multiple segments
        # Extract first frame to get dimensions
        cap = cv2.VideoCapture(str(video_path))
        ret, first_frame = cap.read()
        h, w = first_frame.shape[:2]
        cap.release()
        
        # Create a grid of points to track different regions
        grid_points = []
        grid_size = 4  # 4x4 grid
        for i in range(grid_size):
            for j in range(grid_size):
                x = int((j + 0.5) * w / grid_size)
                y = int((i + 0.5) * h / grid_size)
                grid_points.append([x, y])
        
        # Run SAM2 with multiple points
        print(f"   Tracking {len(grid_points)} regions...")
        
        # Use the working SAM2 video model
        output = replicate.run(
            "meta/sam-2-video:5e0f4fda7fc1b6c63de84378cf30e7cc9b104e6c332c96b3e739763bc46e070a",
            input={
                "video_input": open(str(video_path), "rb"),
                "points": json.dumps(grid_points),
                "labels": json.dumps([1] * len(grid_points)),  # All foreground
                "output_format": "mp4"
            }
        )
        
        # Download segmented video
        import requests
        response = requests.get(output)
        with open(segments_video, 'wb') as f:
            f.write(response.content)
        
        print(f"✅ Segmentation video saved: {segments_video}")
        
        # Extract segment information by analyzing the video
        print("\n📊 Analyzing segments...")
        segments_info = analyze_segments(segments_video, video_path)
        
        # Save metadata
        with open(segments_meta, 'w') as f:
            json.dump(segments_info, f, indent=2)
        
        print(f"✅ Segment metadata saved: {segments_meta}")
        print(f"\n📈 Found {segments_info['num_segments']} semantic regions")
        
        return segments_meta
        
    except Exception as e:
        print(f"❌ SAM2 segmentation failed: {e}")
        return None


def analyze_segments(segments_video_path, original_video_path):
    """
    Analyze the segmented video to extract region information.
    """
    cap_seg = cv2.VideoCapture(str(segments_video_path))
    cap_orig = cv2.VideoCapture(str(original_video_path))
    
    if not cap_seg.isOpened() or not cap_orig.isOpened():
        return {"error": "Could not open videos"}
    
    # Get video properties
    fps = cap_seg.get(cv2.CAP_PROP_FPS)
    width = int(cap_seg.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap_seg.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap_seg.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Sample frames to understand segments
    sample_frames = [0, total_frames//4, total_frames//2, 3*total_frames//4, total_frames-1]
    segments_data = []
    
    for frame_idx in sample_frames:
        cap_seg.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap_seg.read()
        if not ret:
            continue
        
        # Find unique colors (segments)
        # SAM2 typically uses different colors for different segments
        unique_colors = find_unique_segments(frame)
        
        for color in unique_colors:
            # Create mask for this segment
            mask = np.all(np.abs(frame - color) < 30, axis=-1)
            
            if np.sum(mask) < 100:  # Skip tiny segments
                continue
            
            # Find bounding box
            coords = np.where(mask)
            if len(coords[0]) == 0:
                continue
            
            y_min, y_max = coords[0].min(), coords[0].max()
            x_min, x_max = coords[1].min(), coords[1].max()
            
            # Calculate segment properties
            segment_info = {
                "frame": frame_idx,
                "time": frame_idx / fps,
                "color": color.tolist(),
                "bbox": [int(x_min), int(y_min), int(x_max), int(y_max)],
                "area": int(np.sum(mask)),
                "center": [int((x_min + x_max) / 2), int((y_min + y_max) / 2)],
                "position_zone": classify_position_zone(y_min, y_max, height)
            }
            segments_data.append(segment_info)
    
    cap_seg.release()
    cap_orig.release()
    
    # Aggregate segment information
    unique_segments = consolidate_segments(segments_data)
    
    return {
        "video_width": width,
        "video_height": height,
        "fps": fps,
        "total_frames": total_frames,
        "num_segments": len(unique_segments),
        "segments": unique_segments,
        "sampled_frames": sample_frames
    }


def find_unique_segments(frame):
    """Find unique segment colors in frame."""
    # Downsample for faster processing
    small = cv2.resize(frame, (frame.shape[1]//4, frame.shape[0]//4))
    pixels = small.reshape(-1, 3)
    
    # Find unique colors (with some tolerance)
    unique_colors = []
    for color in np.unique(pixels, axis=0):
        # Skip black/white/gray (likely background or edges)
        if np.std(color) < 10:
            continue
        unique_colors.append(color)
    
    return unique_colors[:10]  # Limit to 10 segments


def classify_position_zone(y_min, y_max, height):
    """Classify vertical position zone."""
    y_center = (y_min + y_max) / 2
    
    if y_center < height * 0.33:
        return "top"
    elif y_center < height * 0.66:
        return "middle"
    else:
        return "bottom"


def consolidate_segments(segments_data):
    """Consolidate segments across frames."""
    # Group by similar colors and positions
    consolidated = []
    
    for segment in segments_data:
        # Check if this matches an existing consolidated segment
        matched = False
        for cons in consolidated:
            # Check if colors are similar
            color_diff = np.mean(np.abs(np.array(segment["color"]) - np.array(cons["avg_color"])))
            if color_diff < 30:  # Similar color
                # Update consolidated segment
                cons["occurrences"] += 1
                cons["bboxes"].append(segment["bbox"])
                cons["zones"].add(segment["position_zone"])
                matched = True
                break
        
        if not matched:
            # Create new consolidated segment
            consolidated.append({
                "id": len(consolidated),
                "avg_color": segment["color"],
                "occurrences": 1,
                "bboxes": [segment["bbox"]],
                "zones": {segment["position_zone"]},
                "avg_area": segment["area"]
            })
    
    # Calculate average properties
    for cons in consolidated:
        # Average bounding box
        bboxes = np.array(cons["bboxes"])
        cons["avg_bbox"] = np.mean(bboxes, axis=0).astype(int).tolist()
        cons["zones"] = list(cons["zones"])
        del cons["bboxes"]  # Remove detailed data
    
    return consolidated


if __name__ == "__main__":
    # Test with the ai_math1 video
    video_path = "../../uploads/assets/videos/ai_math1.mp4"
    
    metadata_path = extract_sam2_video_segments(video_path)
    
    if metadata_path:
        # Load and display summary
        with open(metadata_path, 'r') as f:
            data = json.load(f)
        
        print("\n📊 Segment Analysis Summary:")
        print(f"   Video: {data['video_width']}x{data['video_height']}")
        print(f"   Segments found: {data['num_segments']}")
        
        for seg in data['segments']:
            print(f"\n   Segment {seg['id']}:")
            print(f"     Position zones: {seg['zones']}")
            print(f"     Avg bbox: {seg['avg_bbox']}")
            print(f"     Occurrences: {seg['occurrences']}")
