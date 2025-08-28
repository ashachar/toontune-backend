"""
Test multiple 3D animations to verify opacity fix works for all
"""

import cv2
import numpy as np
import sys
import os

# Add animation modules to path
sys.path.append('utils/animations')
sys.path.append('utils/animations/3d_animations')
sys.path.append('utils/animations/3d_animations/opacity_3d')
sys.path.append('utils/animations/3d_animations/motion_3d')
sys.path.append('utils/animations/3d_animations/scale_3d')
sys.path.append('utils/animations/3d_animations/progressive_3d')

from base_3d_text_animation import Animation3DConfig
from opacity_3d import Fade3D, GlowPulse3D, Materialize3D
from motion_3d import Slide3D, Float3D
from scale_3d import Zoom3D
from progressive_3d import Typewriter3D


def test_multiple_3d():
    """Test various 3D animations with the opacity fix"""
    
    print("Testing multiple 3D animations with opacity fix...")
    print("=" * 60)
    
    width, height = 1280, 720
    fps = 30
    duration_seconds = 12  # 3 seconds per animation
    total_frames = fps * duration_seconds
    
    # Create output video
    output_path = "outputs/test_all_3d_fixed.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Test animations
    animations = []
    
    # 1. Fade3D (0-3s)
    config = Animation3DConfig(
        text="FADE 3D",
        duration_ms=2500,
        position=(640, 200, 0),
        font_size=80,
        font_color=(255, 255, 255),
        stagger_ms=30
    )
    animations.append({
        "anim": Fade3D(config, fade_mode="wave"),
        "start": 0,
        "duration": 3
    })
    
    # 2. Zoom3D (3-6s)
    config = Animation3DConfig(
        text="ZOOM 3D",
        duration_ms=2500,
        position=(640, 360, 0),
        font_size=80,
        font_color=(100, 255, 100),
        stagger_ms=30
    )
    animations.append({
        "anim": Zoom3D(config),
        "start": 3,
        "duration": 3
    })
    
    # 3. Typewriter3D (6-9s)
    config = Animation3DConfig(
        text="TYPEWRITER",
        duration_ms=2500,
        position=(640, 500, 0),
        font_size=80,
        font_color=(255, 100, 100),
        stagger_ms=0
    )
    animations.append({
        "anim": Typewriter3D(config),
        "start": 6,
        "duration": 3
    })
    
    # 4. Materialize3D (9-12s)
    config = Animation3DConfig(
        text="MATERIALIZE",
        duration_ms=2500,
        position=(640, 360, 0),
        font_size=80,
        font_color=(100, 100, 255),
        stagger_ms=30
    )
    animations.append({
        "anim": Materialize3D(config),
        "start": 9,
        "duration": 3
    })
    
    print("Testing 4 animations: Fade3D, Zoom3D, Typewriter3D, Materialize3D")
    
    # Render frames
    for frame_num in range(total_frames):
        # Create black background
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Add subtle grid
        for x in range(0, width, 100):
            cv2.line(frame, (x, 0), (x, height), (20, 20, 20), 1)
        for y in range(0, height, 100):
            cv2.line(frame, (0, y), (width, y), (20, 20, 20), 1)
        
        current_time = frame_num / fps
        
        # Apply active animations
        for anim_data in animations:
            anim_start = anim_data["start"]
            anim_duration = anim_data["duration"]
            
            if anim_start <= current_time < anim_start + anim_duration:
                relative_time = current_time - anim_start
                relative_frame = int(relative_time * fps)
                frame = anim_data["anim"].apply_frame(frame, relative_frame, fps)
        
        # Add label for current animation
        if current_time < 3:
            label = "Fade3D"
        elif current_time < 6:
            label = "Zoom3D"
        elif current_time < 9:
            label = "Typewriter3D"
        else:
            label = "Materialize3D"
            
        cv2.putText(frame, f"Testing: {label}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
        cv2.putText(frame, f"Time: {current_time:.1f}s", 
                   (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        out.write(frame)
        
        if frame_num % 30 == 0:
            print(f"  Progress: {current_time:.0f}s / {duration_seconds}s")
    
    out.release()
    
    # Convert to H.264
    h264_output = output_path.replace('.mp4', '_h264.mp4')
    convert_cmd = f"ffmpeg -i {output_path} -c:v libx264 -preset fast -crf 23 -pix_fmt yuv420p -movflags +faststart {h264_output} -y"
    os.system(convert_cmd)
    os.remove(output_path)
    
    print(f"\n✅ Test video created: {h264_output}")
    return h264_output


if __name__ == "__main__":
    print("TESTING ALL 3D ANIMATIONS WITH OPACITY FIX")
    print("=" * 60)
    print()
    
    output = test_multiple_3d()
    
    print("\n" + "=" * 60)
    print("SUCCESS!")
    print("All animations should now have:")
    print("  • Bright, fully visible text")
    print("  • Proper opacity transitions")
    print("  • Solid filled letters")
    print("=" * 60)