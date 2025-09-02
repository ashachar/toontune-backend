#!/usr/bin/env python3
"""
Create SAM2 video segmentation using correct Replicate API.
"""

import os
import replicate
from dotenv import load_dotenv

load_dotenv()

def create_sam2_video_segments(input_video_path):
    """Create SAM2 segmented video with automatic detection."""
    
    print(f"🎯 Creating SAM2 video segmentation for: {input_video_path}")
    
    # Upload video to Replicate (or use URL if available)
    # For local files, you'd need to upload first
    
    # Example with grid of click points for automatic-like segmentation
    # Create a 4x3 grid of points
    width, height = 1280, 720  # Assuming standard dimensions
    points = []
    ids = []
    
    for i in range(3):  # 3 rows
        for j in range(4):  # 4 columns
            x = int((j + 0.5) * width / 4)
            y = int((i + 0.5) * height / 3)
            points.append(f"[{x},{y}]")
            ids.append(f"seg_{i*4+j}")
    
    click_coordinates = ",".join(points)
    click_object_ids = ",".join(ids)
    
    input_config = {
        "mask_type": "highlighted",
        "video_fps": 25,
        "input_video": open(input_video_path, "rb"),
        "click_frames": "1",  # Use first frame
        "output_video": True,
        "click_object_ids": click_object_ids,
        "click_coordinates": click_coordinates
    }
    
    print(f"   Using {len(points)} click points for segmentation")
    
    # Run SAM2 video
    output = replicate.run(
        "meta/sam-2-video:33432afdfc06a10da6b4018932893d39b0159f838b6d11dd1236dff85cc5ec1d",
        input=input_config
    )
    
    print(f"✅ SAM2 segmentation complete: {output}")
    return output

if __name__ == "__main__":
    # Test with 10 second clip
    import subprocess
    
    test_video = "/tmp/test_10s.mp4"
    subprocess.run([
        "ffmpeg", "-i", "../../uploads/assets/videos/ai_math1.mp4",
        "-t", "10", "-c:v", "libx264", "-y", test_video
    ], capture_output=True)
    
    result = create_sam2_video_segments(test_video)
    print(f"Result: {result}")
