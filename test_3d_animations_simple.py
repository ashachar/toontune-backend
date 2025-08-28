"""
Simple test for 3D text animations - tests a subset to verify functionality
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
sys.path.append('utils/animations/3d_animations/motion_3d')
sys.path.append('utils/animations/3d_animations/scale_3d')

from base_3d_text_animation import Animation3DConfig
from opacity_3d import Fade3D, Dissolve3D
from motion_3d import Float3D, Orbit3D
from scale_3d import Zoom3D, Rotate3DAxis


def test_3d_animations_simple():
    """Test a few key 3D animations"""
    
    print("Creating test video with 3D animations...")
    
    # Create a blank video
    width, height = 1280, 720
    fps = 30
    duration_seconds = 20
    total_frames = fps * duration_seconds
    
    # Create output video
    output_path = "outputs/3d_animations_simple_test.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Create animations
    animations = []
    
    # 1. 3D Fade
    config = Animation3DConfig(
        text="3D FADE",
        duration_ms=3000,
        position=(640, 200, 0),
        font_size=60,
        font_color=(255, 255, 255),
        depth_color=(180, 180, 180),
        stagger_ms=50
    )
    animations.append({
        "anim": Fade3D(config, fade_mode="wave"),
        "start": 0,
        "duration": 3
    })
    
    # 2. 3D Float
    config = Animation3DConfig(
        text="FLOATING",
        duration_ms=3000,
        position=(640, 300, 0),
        font_size=60,
        font_color=(255, 200, 100)
    )
    animations.append({
        "anim": Float3D(config, float_pattern="wave"),
        "start": 3.5,
        "duration": 3
    })
    
    # 3. 3D Zoom
    config = Animation3DConfig(
        text="ZOOM 3D",
        duration_ms=3000,
        position=(640, 400, 0),
        font_size=70,
        font_color=(255, 100, 100)
    )
    animations.append({
        "anim": Zoom3D(config, spiral_zoom=False, pulsate=False),
        "start": 7,
        "duration": 3
    })
    
    # 4. 3D Rotate
    config = Animation3DConfig(
        text="ROTATE",
        duration_ms=3000,
        position=(640, 500, 0),
        font_size=60,
        font_color=(100, 100, 255)
    )
    animations.append({
        "anim": Rotate3DAxis(config, rotation_axis="y"),
        "start": 10.5,
        "duration": 3
    })
    
    # 5. 3D Dissolve
    config = Animation3DConfig(
        text="DISSOLVE",
        duration_ms=3000,
        position=(640, 360, 0),
        font_size=65,
        font_color=(200, 255, 200)
    )
    animations.append({
        "anim": Dissolve3D(config, dissolve_direction="up"),
        "start": 14,
        "duration": 3
    })
    
    # Render frames
    for frame_num in range(total_frames):
        # Create black background
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Add grid for depth perception
        for x in range(0, width, 50):
            cv2.line(frame, (x, 0), (x, height), (30, 30, 30), 1)
        for y in range(0, height, 50):
            cv2.line(frame, (0, y), (width, y), (30, 30, 30), 1)
        
        current_time = frame_num / fps
        
        # Apply animations
        for anim_data in animations:
            anim_start = anim_data["start"]
            anim_duration = anim_data["duration"]
            
            if anim_start <= current_time < anim_start + anim_duration:
                relative_time = current_time - anim_start
                relative_frame = int(relative_time * fps)
                frame = anim_data["anim"].apply_frame(frame, relative_frame, fps)
        
        # Add title
        cv2.putText(frame, "3D Text Animation Test", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Add time
        cv2.putText(frame, f"Time: {current_time:.1f}s", 
                   (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        out.write(frame)
        
        if frame_num % fps == 0:
            print(f"  Progress: {current_time:.0f}s / {duration_seconds}s")
    
    out.release()
    
    # Convert to H.264
    h264_output = output_path.replace('.mp4', '_h264.mp4')
    convert_cmd = f"ffmpeg -i {output_path} -c:v libx264 -preset fast -crf 23 -pix_fmt yuv420p -movflags +faststart {h264_output} -y"
    os.system(convert_cmd)
    os.remove(output_path)
    
    print(f"\n✅ Test complete: {h264_output}")
    print("\nAnimations tested:")
    print("1. 3D Fade with wave effect")
    print("2. 3D Float with bobbing motion")
    print("3. 3D Zoom from depth")
    print("4. 3D Rotation around Y-axis")
    print("5. 3D Dissolve upward")
    
    return h264_output


if __name__ == "__main__":
    print("=" * 60)
    print("SIMPLE 3D TEXT ANIMATIONS TEST")
    print("=" * 60)
    print()
    
    output = test_3d_animations_simple()
    
    print("\n" + "=" * 60)
    print("SUCCESS!")
    print("=" * 60)