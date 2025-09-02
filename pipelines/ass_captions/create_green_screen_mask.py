#!/usr/bin/env python3
"""
Create a green screen mask video for testing purposes.
This is a simple placeholder that creates a green video the same size as the input.
"""

import cv2
import numpy as np
import sys

def create_green_screen_mask(input_video, output_path):
    """Create a green screen mask video."""
    
    # Open input video to get properties
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print(f"Error: Cannot open {input_video}")
        return False
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Creating green screen mask: {width}x{height}, {fps} fps, {total_frames} frames")
    
    # Create output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Green screen color (the specific green used by RVM)
    green_color = np.array([119, 254, 154], dtype=np.uint8)  # BGR format
    
    # Create frames
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Create a fully green frame
        green_frame = np.full((height, width, 3), green_color, dtype=np.uint8)
        
        # Optional: Keep the person area (very basic - just center region)
        # This is a placeholder - in reality RVM would do proper segmentation
        center_x, center_y = width // 2, height // 2
        radius = min(width, height) // 4
        
        # Create a mask for the "person" area
        y, x = np.ogrid[:height, :width]
        mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
        
        # Copy original frame content where the "person" would be
        green_frame[mask] = frame[mask]
        
        out.write(green_frame)
        frame_count += 1
        
        if frame_count % 100 == 0:
            print(f"  Processed {frame_count}/{total_frames} frames...")
    
    cap.release()
    out.release()
    
    print(f"✅ Green screen mask created: {output_path}")
    print(f"   Total frames: {frame_count}")
    return True


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python create_green_screen_mask.py <input_video> <output_mask>")
        sys.exit(1)
    
    input_video = sys.argv[1]
    output_mask = sys.argv[2]
    
    if create_green_screen_mask(input_video, output_mask):
        sys.exit(0)
    else:
        sys.exit(1)