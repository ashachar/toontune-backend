"""
Test the improved 3D text rendering with more substantial filling
"""

import cv2
import numpy as np
import sys
import os
from pathlib import Path

# Add animation modules to path
sys.path.append('utils/animations')
sys.path.append('utils/animations/3d_animations')
sys.path.append('utils/animations/3d_animations/opacity_3d')

from base_3d_text_animation import Animation3DConfig
from opacity_3d import Fade3D


def test_substantial_text():
    """Test the improved text rendering"""
    
    print("Testing improved 3D text rendering with substantial filling...")
    print("=" * 60)
    
    # Create a short test video
    width, height = 1280, 720
    fps = 30
    duration_seconds = 5
    total_frames = fps * duration_seconds
    
    # Create output video
    output_path = "outputs/test_substantial_text.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Create animation with different thickness settings
    configs = [
        # Original (sparse)
        Animation3DConfig(
            text="SPARSE",
            duration_ms=4000,
            position=(320, 360, 0),
            font_size=90,
            font_thickness=1,  # Thin
            font_color=(255, 100, 100),
            depth_color=(200, 80, 80),
            stagger_ms=30
        ),
        # Improved (substantial)
        Animation3DConfig(
            text="SUBSTANTIAL",
            duration_ms=4000,
            position=(960, 360, 0),
            font_size=90,
            font_thickness=5,  # Thick
            font_color=(100, 255, 100),
            depth_color=(80, 200, 80),
            stagger_ms=30
        )
    ]
    
    animations = [
        Fade3D(configs[0], fade_mode="wave"),
        Fade3D(configs[1], fade_mode="wave")
    ]
    
    # Render frames
    for frame_num in range(total_frames):
        # Create black background
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Add grid
        for x in range(0, width, 50):
            cv2.line(frame, (x, 0), (x, height), (20, 20, 20), 1)
        for y in range(0, height, 50):
            cv2.line(frame, (0, y), (width, y), (20, 20, 20), 1)
        
        # Apply both animations
        for anim in animations:
            frame = anim.apply_frame(frame, frame_num, fps)
        
        # Add labels
        cv2.putText(frame, "Before (Sparse)", 
                   (200, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, "After (Substantial)", 
                   (850, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add dividing line
        cv2.line(frame, (640, 100), (640, 620), (100, 100, 100), 2)
        
        out.write(frame)
        
        if frame_num % 30 == 0:
            print(f"  Progress: {frame_num}/{total_frames} frames")
    
    out.release()
    
    # Convert to H.264
    h264_output = output_path.replace('.mp4', '_h264.mp4')
    convert_cmd = f"ffmpeg -i {output_path} -c:v libx264 -preset fast -crf 23 -pix_fmt yuv420p -movflags +faststart {h264_output} -y"
    os.system(convert_cmd)
    os.remove(output_path)
    
    print(f"\n✅ Test video created: {h264_output}")
    
    # Extract comparison frame
    cap = cv2.VideoCapture(h264_output)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 60)  # Frame at 2 seconds
    ret, frame = cap.read()
    if ret:
        cv2.imwrite("outputs/text_comparison.jpg", frame)
        print("Comparison frame saved: outputs/text_comparison.jpg")
    cap.release()
    
    return h264_output


if __name__ == "__main__":
    print("3D TEXT SUBSTANTIAL RENDERING TEST")
    print("=" * 60)
    print()
    
    output = test_substantial_text()
    
    print("\n" + "=" * 60)
    print("TEST COMPLETE!")
    print("The video shows:")
    print("  - LEFT: Sparse text (thin stroke)")
    print("  - RIGHT: Substantial text (thick stroke)")
    print("=" * 60)