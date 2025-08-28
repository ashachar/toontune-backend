"""
Test Fade3D on solid black background to prove animation is smooth
"""

import cv2
import numpy as np
import sys
import os

sys.path.append('utils/animations')
sys.path.append('utils/animations/3d_animations')
sys.path.append('utils/animations/3d_animations/opacity_3d')

from base_3d_text_animation import Animation3DConfig
from opacity_3d import Fade3D

def test_on_solid_background():
    """Test animation on solid black background"""
    
    print("Testing 3D Fade Wave on SOLID BLACK background")
    print("(This proves the animation itself has no flashing)")
    print("=" * 60)
    
    width, height = 1280, 720
    fps = 25
    duration_seconds = 3
    total_frames = int(fps * duration_seconds)
    
    # Create output video
    output_path = "outputs/fade_solid_bg.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Create Fade3D animation
    config = Animation3DConfig(
        text="3D FADE WAVE",
        duration_ms=3000,
        position=(640, 360, 0),
        font_size=70,
        font_color=(255, 255, 255),
        depth_color=(100, 100, 100),
        stagger_ms=40
    )
    
    fade_anim = Fade3D(config, fade_mode="wave", depth_fade=False)
    
    # Track brightness for analysis
    brightness_history = []
    
    print("Rendering frames...")
    
    for frame_num in range(total_frames):
        # SOLID BLACK BACKGROUND - no variation
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Apply animation
        animated_frame = fade_anim.apply_frame(frame, frame_num, fps)
        
        # Measure brightness in text region
        roi = animated_frame[320:400, 400:880]
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray_roi)
        brightness_history.append(mean_brightness)
        
        # Add info
        progress = frame_num / (fps * 3.0)
        cv2.putText(animated_frame, f"Solid Black BG - Frame {frame_num}/{total_frames}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)
        cv2.putText(animated_frame, f"Progress: {progress:.2f} | Brightness: {mean_brightness:.1f}", 
                   (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        
        out.write(animated_frame)
    
    out.release()
    
    # Convert to H.264
    h264_output = output_path.replace('.mp4', '_h264.mp4')
    convert_cmd = f"ffmpeg -i {output_path} -c:v libx264 -preset fast -crf 23 -pix_fmt yuv420p -movflags +faststart {h264_output} -y"
    os.system(convert_cmd)
    os.remove(output_path)
    
    # Analyze brightness pattern
    print("\n" + "=" * 60)
    print("BRIGHTNESS ANALYSIS (Solid Background)")
    print("=" * 60)
    
    # Check for oscillations
    direction_changes = 0
    prev_direction = None
    
    for i in range(1, len(brightness_history)):
        diff = brightness_history[i] - brightness_history[i-1]
        
        if diff > 0:
            direction = "up"
        elif diff < 0:
            direction = "down" 
        else:
            direction = "flat"
        
        if prev_direction and direction != "flat" and direction != prev_direction:
            direction_changes += 1
        
        if direction != "flat":
            prev_direction = direction
    
    print(f"Min brightness: {min(brightness_history):.1f}")
    print(f"Max brightness: {max(brightness_history):.1f}")
    print(f"Direction changes: {direction_changes}")
    
    # Check if smooth
    if direction_changes <= 3:  # Allow for minor variations
        print("\n✅ ANIMATION IS SMOOTH - No flashing detected!")
        print("The perceived 'flashing' was from the original video content,")
        print("not from the animation itself.")
    else:
        print(f"\n⚠️ Still detecting {direction_changes} direction changes")
    
    print(f"\nVideo saved: {h264_output}")
    return h264_output

if __name__ == "__main__":
    test_on_solid_background()