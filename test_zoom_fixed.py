"""
Test Zoom3D with artifact fix
"""

import cv2
import numpy as np
import sys
import os

# Add animation modules to path
sys.path.append('utils/animations')
sys.path.append('utils/animations/3d_animations')
sys.path.append('utils/animations/3d_animations/scale_3d')

from base_3d_text_animation import Animation3DConfig
from scale_3d import Zoom3D


def test_zoom_fixed():
    """Test Zoom3D with the artifact fix"""
    
    print("Testing Zoom3D with artifact fix...")
    print("=" * 60)
    
    width, height = 1280, 720
    fps = 30
    duration_seconds = 4
    total_frames = fps * duration_seconds
    
    # Create output video
    output_path = "outputs/zoom_fixed.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Create Zoom animation with green text to make artifacts more visible
    config = Animation3DConfig(
        text="ZOOM 3D",
        duration_ms=3500,
        position=(640, 360, 0),
        font_size=100,
        font_color=(100, 255, 100),  # Green like in the original test
        depth_color=(50, 150, 50),
        stagger_ms=30
    )
    
    zoom_anim = Zoom3D(config, start_scale=0.1, end_scale=1.0)
    
    print("Rendering frames...")
    
    # Render frames
    for frame_num in range(total_frames):
        # Black background
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Add subtle grid
        for x in range(0, width, 100):
            cv2.line(frame, (x, 0), (x, height), (20, 20, 20), 1)
        for y in range(0, height, 100):
            cv2.line(frame, (0, y), (width, y), (20, 20, 20), 1)
        
        # Apply animation
        frame = zoom_anim.apply_frame(frame, frame_num, fps)
        
        # Add info
        progress = min(1.0, frame_num / (fps * 3.5))
        scale = 0.1 + (1.0 - 0.1) * progress
        cv2.putText(frame, f"Zoom3D Test - Scale: {scale:.2f}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
        cv2.putText(frame, "Fixed: Alpha premultiplication + INTER_AREA", 
                   (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        out.write(frame)
        
        if frame_num % 30 == 0:
            print(f"  Progress: {frame_num}/{total_frames} frames")
        
        # Save key frames for comparison
        if frame_num in [15, 30, 60, 90]:
            cv2.imwrite(f"outputs/zoom_fixed_frame_{frame_num}.jpg", frame)
    
    out.release()
    
    # Convert to H.264
    h264_output = output_path.replace('.mp4', '_h264.mp4')
    convert_cmd = f"ffmpeg -i {output_path} -c:v libx264 -preset fast -crf 23 -pix_fmt yuv420p -movflags +faststart {h264_output} -y"
    os.system(convert_cmd)
    os.remove(output_path)
    
    print(f"\n✅ Fixed video created: {h264_output}")
    
    # Analyze a frame for artifacts
    test_frame = cv2.imread("outputs/zoom_fixed_frame_30.jpg")
    if test_frame is not None:
        # Check for white pixels (high brightness in all channels)
        gray = cv2.cvtColor(test_frame, cv2.COLOR_BGR2GRAY)
        white_pixels = np.sum((gray > 200) & (gray < 255))  # Exclude pure white text
        
        print(f"\nArtifact analysis (frame 30):")
        print(f"  Suspicious white pixels: {white_pixels}")
        if white_pixels < 50:
            print("  ✅ Artifacts significantly reduced!")
        else:
            print("  ⚠️ Some artifacts may still be present")
    
    return h264_output


if __name__ == "__main__":
    print("ZOOM3D ARTIFACT FIX TEST")
    print("=" * 60)
    print()
    
    output = test_zoom_fixed()
    
    print("\n" + "=" * 60)
    print("FIX APPLIED!")
    print("The fix uses:")
    print("  • Alpha premultiplication before scaling")
    print("  • INTER_AREA for downscaling (better quality)")
    print("  • Proper alpha channel handling")
    print("=" * 60)