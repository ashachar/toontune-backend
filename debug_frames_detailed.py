"""
Debug specific frames to find source of brightness jumps
"""

import cv2
import numpy as np
import sys
sys.path.append('utils/animations')
sys.path.append('utils/animations/3d_animations')
sys.path.append('utils/animations/3d_animations/opacity_3d')

from base_3d_text_animation import Animation3DConfig
from opacity_3d import Fade3D

def debug_specific_frames():
    """Debug frames 60-75 where brightness jumps occur"""
    
    print("Debugging frames 60-75 for brightness jumps")
    print("=" * 60)
    
    # Load video
    input_video = "uploads/assets/videos/ai_math1.mp4"
    cap = cv2.VideoCapture(input_video)
    
    if not cap.isOpened():
        print(f"Error: Could not open {input_video}")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Create Fade3D animation with same config
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
    
    # Focus on frames 60-75
    start_frame = 60
    end_frame = 75
    
    # Seek to start frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    print("Frame | Progress | Letter Opacities | Mean Brightness")
    print("-" * 60)
    
    for frame_num in range(start_frame, end_frame + 1):
        ret, frame = cap.read()
        if not ret:
            break
        
        # Apply animation
        animated_frame = fade_anim.apply_frame(frame, frame_num, fps)
        
        # Calculate progress
        progress = frame_num / (fps * 3.0)
        
        # Get letter opacities
        opacities = [f"{letter.opacity:.2f}" for letter in fade_anim.letters[:3]]
        
        # Calculate brightness in text region
        roi = animated_frame[320:400, 400:880]
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray_roi)
        
        # Check if animation should be complete
        if progress >= 1.0:
            status = "COMPLETE"
        else:
            status = "ANIMATING"
        
        print(f"{frame_num:3d} | {progress:.3f} | {' '.join(opacities)} | {mean_brightness:.1f} | {status}")
        
        # Check for issue at frame 65-70
        if frame_num == 65:
            print("  -> Frame 65: Check if animation restarts or loops")
        if frame_num == 70:
            print("  -> Frame 70: Brightness drops - why?")
    
    cap.release()
    
    # Additional check: what happens when progress > 1.0?
    print("\n" + "=" * 60)
    print("Testing progress > 1.0 behavior:")
    
    test_progresses = [0.9, 0.95, 1.0, 1.05, 1.1]
    for prog in test_progresses:
        # Update animation
        fade_anim.update_letters(prog, 0, fps)
        opacities = [letter.opacity for letter in fade_anim.letters[:3]]
        print(f"Progress {prog:.2f}: opacities = {[f'{o:.3f}' for o in opacities]}")

if __name__ == "__main__":
    debug_specific_frames()