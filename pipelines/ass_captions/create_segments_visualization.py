#!/usr/bin/env python3
"""
Create a visualization of video segments using color-based segmentation.
This shows different regions of the video that could be used for text placement.
"""

import cv2
import numpy as np
from pathlib import Path
from sklearn.cluster import KMeans

def create_segments_visualization(input_video: str, output_path: str = None):
    """
    Create a video showing segmented regions.
    """
    video_path = Path(input_video)
    if not video_path.exists():
        print(f"Error: Video not found: {video_path}")
        return None
    
    if output_path is None:
        output_path = f"../../outputs/{video_path.stem}_segments_viz.mp4"
    
    print(f"\n🎨 Creating segments visualization for: {input_video}")
    
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print("Error: Cannot open video")
        return None
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width * 2, height))  # Side-by-side
    
    print(f"   Video: {width}x{height}, {total_frames} frames")
    print(f"   Creating side-by-side comparison (original | segments)")
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Create segmentation
        segmented = create_frame_segments(frame)
        
        # Add labels
        cv2.putText(frame, "Original", (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(segmented, "Segments", (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Add segment regions overlay
        overlay = create_region_overlay(segmented)
        segmented = cv2.addWeighted(segmented, 0.7, overlay, 0.3, 0)
        
        # Combine side by side
        combined = np.hstack([frame, segmented])
        out.write(combined)
        
        if frame_idx % 100 == 0:
            print(f"      Processed {frame_idx}/{total_frames} frames")
        
        frame_idx += 1
    
    cap.release()
    out.release()
    
    print(f"\n✅ Segments visualization created: {output_path}")
    
    # Convert to H.264
    output_h264 = output_path.replace('.mp4', '_h264.mp4')
    import subprocess
    cmd = [
        'ffmpeg', '-i', output_path,
        '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
        '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
        '-y', output_h264
    ]
    subprocess.run(cmd, capture_output=True)
    
    # Remove temp file
    import os
    os.remove(output_path)
    
    print(f"✅ H.264 version: {output_h264}")
    return output_h264


def create_frame_segments(frame, n_segments=8):
    """
    Create segmented version of frame using K-means clustering.
    """
    # Downsample for faster processing
    small = cv2.resize(frame, (frame.shape[1]//4, frame.shape[0]//4))
    
    # Reshape for clustering
    pixels = small.reshape(-1, 3)
    
    # K-means clustering
    kmeans = KMeans(n_clusters=n_segments, random_state=42, n_init=10)
    labels = kmeans.fit_predict(pixels)
    
    # Create segmented image
    centers = kmeans.cluster_centers_.astype(np.uint8)
    segmented_small = centers[labels].reshape(small.shape)
    
    # Resize back
    segmented = cv2.resize(segmented_small, (frame.shape[1], frame.shape[0]), 
                           interpolation=cv2.INTER_NEAREST)
    
    return segmented


def create_region_overlay(segmented_frame):
    """
    Create overlay showing distinct regions.
    """
    h, w = segmented_frame.shape[:2]
    overlay = np.zeros_like(segmented_frame)
    
    # Define regions for text placement
    regions = [
        # Top regions
        {"name": "top-left", "coords": (0, 0, w//3, h//3), "color": (255, 100, 100)},
        {"name": "top-center", "coords": (w//3, 0, 2*w//3, h//3), "color": (100, 255, 100)},
        {"name": "top-right", "coords": (2*w//3, 0, w, h//3), "color": (100, 100, 255)},
        
        # Middle regions
        {"name": "mid-left", "coords": (0, h//3, w//3, 2*h//3), "color": (255, 255, 100)},
        {"name": "mid-center", "coords": (w//3, h//3, 2*w//3, 2*h//3), "color": (255, 100, 255)},
        {"name": "mid-right", "coords": (2*w//3, h//3, w, 2*h//3), "color": (100, 255, 255)},
        
        # Bottom regions
        {"name": "bottom-left", "coords": (0, 2*h//3, w//3, h), "color": (200, 150, 100)},
        {"name": "bottom-center", "coords": (w//3, 2*h//3, 2*w//3, h), "color": (100, 200, 150)},
        {"name": "bottom-right", "coords": (2*w//3, 2*h//3, w, h), "color": (150, 100, 200)},
    ]
    
    # Draw region boundaries
    for region in regions:
        x1, y1, x2, y2 = region["coords"]
        # Draw rectangle border
        cv2.rectangle(overlay, (x1, y1), (x2, y2), region["color"], 2)
        # Add semi-transparent fill
        roi = overlay[y1:y2, x1:x2]
        roi[:, :] = roi[:, :] * 0.7 + np.array(region["color"]) * 0.3
    
    return overlay


if __name__ == "__main__":
    # Create visualization for first 30 seconds
    import subprocess
    import os
    
    print("Creating 30-second test video...")
    test_input = "../../uploads/assets/videos/ai_math1.mp4"
    test_30s = "/tmp/ai_math1_30s.mp4"
    
    # Extract 30 seconds
    subprocess.run([
        'ffmpeg', '-i', test_input,
        '-t', '30', '-c:v', 'libx264', '-preset', 'fast',
        '-y', test_30s
    ], capture_output=True)
    
    # Create segments visualization
    output = create_segments_visualization(test_30s)
    
    if output:
        print(f"\n✨ Segments visualization ready: {output}")
        print("   Left: Original video")
        print("   Right: Segmented regions with overlay grid")
        print("   The grid shows 9 regions where text can be placed")
    
    # Clean up
    if os.path.exists(test_30s):
        os.remove(test_30s)
