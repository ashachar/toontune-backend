#!/usr/bin/env python3
"""
Debug version to fix coordinate issues in drawing pipeline
"""

import cv2
import numpy as np
from skimage.morphology import skeletonize
import os
import sys

def test_simple_drawing(skeleton_path, output_video):
    """Test drawing with proper coordinate handling"""
    print("Testing simple drawing...")
    
    # Read skeleton
    img = cv2.imread(skeleton_path, cv2.IMREAD_GRAYSCALE)
    height, width = img.shape
    
    # Get all black pixels (lines)
    lines = (img < 128).astype(np.uint8) * 255
    
    # Find connected components
    num_labels, labels = cv2.connectedComponents(lines, connectivity=8)
    print(f"Found {num_labels - 1} components")
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 30
    video_writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    
    if not video_writer.isOpened():
        print("Failed to open video writer")
        return False
    
    # Create canvas
    canvas = np.ones((height, width, 3), dtype=np.uint8) * 255
    
    # Process each component
    for label in range(1, min(num_labels, 6)):  # Test first 5 components
        print(f"Drawing component {label}")
        
        # Get all pixels for this component (y, x format from numpy)
        points = np.argwhere(labels == label)
        
        if len(points) < 10:
            continue
        
        print(f"  Component {label}: {len(points)} pixels")
        print(f"  First few points (y,x): {points[:5].tolist()}")
        
        # Sort points for simple top-to-bottom, left-to-right drawing
        sorted_indices = np.lexsort((points[:, 1], points[:, 0]))  # Sort by y, then x
        sorted_points = points[sorted_indices]
        
        # Draw this component pixel by pixel
        points_per_frame = 50
        
        for i in range(0, len(sorted_points), points_per_frame):
            frame = canvas.copy()
            
            # Draw pixels in this batch
            batch = sorted_points[i:i+points_per_frame]
            
            for y, x in batch:
                # IMPORTANT: OpenCV uses (x, y) for drawing, numpy gives us (y, x)
                canvas[y, x] = [0, 0, 0]  # Draw black pixel
                frame[y, x] = [255, 0, 0]  # Highlight in red on current frame
            
            # Add marker for current position
            if i + points_per_frame < len(sorted_points):
                last_y, last_x = sorted_points[min(i + points_per_frame, len(sorted_points)-1)]
                cv2.circle(frame, (last_x, last_y), 5, (0, 0, 255), -1)
            
            # Add text
            cv2.putText(frame, f"Component {label}/{num_labels-1}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            cv2.putText(frame, f"Pixels: {i}/{len(sorted_points)}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            
            video_writer.write(frame)
    
    # Final frames
    for _ in range(fps):
        final = canvas.copy()
        cv2.putText(final, "Complete!", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 0), 2)
        video_writer.write(final)
    
    video_writer.release()
    print(f"Video saved: {output_video}")
    
    # Convert to H.264
    h264_path = output_video.replace('.mp4', '_h264.mp4')
    os.system(f'ffmpeg -i "{output_video}" -c:v libx264 -preset slow -crf 22 -pix_fmt yuv420p -movflags +faststart "{h264_path}" -y')
    
    return h264_path

if __name__ == "__main__":
    skeleton_path = "robot_euler_output/robot_skeleton.png"
    output_video = "robot_euler_output/robot_debug_drawing.mp4"
    
    h264_video = test_simple_drawing(skeleton_path, output_video)
    
    if h264_video and os.path.exists(h264_video):
        print(f"Opening video: {h264_video}")
        os.system(f'open "{h264_video}"')