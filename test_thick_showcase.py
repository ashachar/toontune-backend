"""
Quick test of 3D animations with substantial text thickness
"""

import cv2
import numpy as np
import sys
import os

# Add animation modules to path
sys.path.append('utils/animations')
sys.path.append('utils/animations/3d_animations')
sys.path.append('utils/animations/3d_animations/opacity_3d')

from base_3d_text_animation import Animation3DConfig
from opacity_3d import Fade3D, BlurFade3D, GlowPulse3D


def test_thick_showcase():
    """Test a few animations with thick text"""
    
    print("Creating test with substantial text...")
    
    # Read first 10 seconds from original video as background
    input_video = "uploads/assets/videos/ai_math1.mp4"
    cap = cv2.VideoCapture(input_video)
    
    if not cap.isOpened():
        print("Error: Could not open input video")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Create output video
    output_path = "outputs/thick_showcase_test.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    position_3d = (640, 360, 0)
    duration_ms = 3000
    
    # Create animations with explicit thick text
    animations = []
    
    # Animation 1: 3D Fade Wave (0-3s)
    config = Animation3DConfig(
        text="3D FADE WAVE",
        duration_ms=duration_ms,
        position=position_3d,
        font_size=80,  # Larger
        font_color=(255, 255, 255),
        font_thickness=5,  # Explicitly thick
        depth_layers=10,
        depth_offset=2,
        stagger_ms=40
    )
    animations.append({
        "animation": Fade3D(config, fade_mode="wave"),
        "start": 0,
        "duration": 3
    })
    
    # Animation 2: 3D Blur Fade (3-6s)
    config = Animation3DConfig(
        text="3D BLUR FADE",
        duration_ms=duration_ms,
        position=position_3d,
        font_size=80,
        font_color=(100, 200, 255),
        font_thickness=5,  # Explicitly thick
        depth_layers=10,
        stagger_ms=40
    )
    animations.append({
        "animation": BlurFade3D(config, start_blur=25, end_blur=0),
        "start": 3,
        "duration": 3
    })
    
    # Animation 3: 3D Glow Pulse (6-9s)
    config = Animation3DConfig(
        text="3D GLOW PULSE",
        duration_ms=duration_ms,
        position=position_3d,
        font_size=80,
        font_color=(255, 200, 100),
        font_thickness=5,  # Explicitly thick
        stagger_ms=40
    )
    animations.append({
        "animation": GlowPulse3D(config, glow_radius=10, pulse_count=2),
        "start": 6,
        "duration": 3
    })
    
    # Process frames
    frame_count = 0
    total_frames = int(9 * fps)  # 9 seconds
    
    print(f"Processing {total_frames} frames...")
    
    while frame_count < total_frames:
        ret, frame = cap.read()
        if not ret:
            # If video ends, use black frame
            frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        current_time = frame_count / fps
        
        # Apply active animations
        for anim_data in animations:
            anim_start = anim_data["start"]
            anim_duration = anim_data["duration"]
            
            if anim_start <= current_time < anim_start + anim_duration:
                relative_time = current_time - anim_start
                relative_frame = int(relative_time * fps)
                
                # Apply animation
                frame = anim_data["animation"].apply_frame(frame, relative_frame, fps)
        
        out.write(frame)
        frame_count += 1
        
        if frame_count % int(fps) == 0:
            print(f"  Progress: {frame_count}/{total_frames} frames ({current_time:.1f}s)")
    
    cap.release()
    out.release()
    
    # Convert to H.264
    h264_output = output_path.replace('.mp4', '_h264.mp4')
    convert_cmd = f"ffmpeg -i {output_path} -c:v libx264 -preset fast -crf 23 -pix_fmt yuv420p -movflags +faststart {h264_output} -y"
    os.system(convert_cmd)
    os.remove(output_path)
    
    print(f"\n✅ Test video created: {h264_output}")
    
    # Extract a frame to show the improvement
    cap = cv2.VideoCapture(h264_output)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(1.5 * fps))  # Frame at 1.5s
    ret, frame = cap.read()
    if ret:
        cv2.imwrite("outputs/thick_text_sample.jpg", frame)
        print("Sample frame saved: outputs/thick_text_sample.jpg")
    cap.release()
    
    return h264_output


if __name__ == "__main__":
    print("=" * 60)
    print("3D TEXT WITH SUBSTANTIAL THICKNESS TEST")
    print("=" * 60)
    print()
    
    output = test_thick_showcase()
    
    print("\n" + "=" * 60)
    print("SUCCESS!")
    print("The video shows 3D text animations with:")
    print("  • Thicker stroke width (5 pixels)")
    print("  • Larger font size (80 pixels)")
    print("  • More depth layers (10 layers)")
    print("  • Better visibility and substance")
    print("=" * 60)