#!/usr/bin/env python3
"""
Create side-by-side comparison of optimal vs center text positioning.
"""

import cv2
import numpy as np
import sys

def create_comparison(video1_path, video2_path, output_path, label1="Optimal", label2="Center"):
    """Create side-by-side comparison video."""
    
    # Open videos
    cap1 = cv2.VideoCapture(video1_path)
    cap2 = cv2.VideoCapture(video2_path)
    
    if not cap1.isOpened() or not cap2.isOpened():
        print("Error opening videos")
        return False
    
    # Get properties from first video
    fps = int(cap1.get(cv2.CAP_PROP_FPS))
    width = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Create output video (double width)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width * 2, height))
    
    frame_count = 0
    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        
        if not ret1 or not ret2:
            break
        
        # Create side-by-side frame
        combined = np.hstack([frame1, frame2])
        
        # Add labels
        cv2.putText(combined, label1, (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
        cv2.putText(combined, label2, (width + 20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)
        
        # Add dividing line
        cv2.line(combined, (width, 0), (width, height), (255, 255, 255), 2)
        
        out.write(combined)
        frame_count += 1
        
        if frame_count % 30 == 0:
            print(f"Processed {frame_count} frames")
    
    # Release everything
    cap1.release()
    cap2.release()
    out.release()
    
    print(f"Comparison video saved to {output_path}")
    print(f"Total frames: {frame_count}")
    
    # Convert to H.264
    import subprocess
    h264_path = output_path.replace('.mp4', '_h264.mp4')
    subprocess.run([
        'ffmpeg', '-i', output_path,
        '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
        '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
        h264_path, '-y'
    ], capture_output=True)
    
    print(f"H.264 version saved to {h264_path}")
    return True

if __name__ == "__main__":
    if len(sys.argv) > 2:
        create_comparison(sys.argv[1], sys.argv[2], 
                         sys.argv[3] if len(sys.argv) > 3 else "comparison.mp4")
    else:
        # Default comparison
        create_comparison(
            "ai_math1_optimal_hq.mp4",
            "ai_math1_center_hq.mp4",
            "text_position_comparison.mp4",
            "Optimal (693, 106)",
            "Center (640, 360)"
        )