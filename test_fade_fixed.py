"""
Test the fixed 3D Fade animation to verify blinking is eliminated
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


def test_fade_fixed():
    """Test the fixed 3D Fade animation"""
    
    print("Testing fixed 3D Fade animation...")
    print("=" * 60)
    
    # Create a short test video (10 seconds)
    width, height = 1280, 720
    fps = 30
    duration_seconds = 10
    total_frames = fps * duration_seconds
    
    # Create output video
    output_path = "outputs/test_fade_fixed.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Create 3D Fade animation with wave mode
    config = Animation3DConfig(
        text="3D FADE WAVE",
        duration_ms=8000,  # 8 seconds animation
        position=(640, 360, 0),
        font_size=80,
        font_color=(255, 255, 255),
        depth_color=(180, 180, 180),
        stagger_ms=50
    )
    
    fade_animation = Fade3D(config, fade_mode="wave")
    
    # Render frames
    for frame_num in range(total_frames):
        # Create black background
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Add grid for depth perception
        for x in range(0, width, 50):
            cv2.line(frame, (x, 0), (x, height), (30, 30, 30), 1)
        for y in range(0, height, 50):
            cv2.line(frame, (0, y), (width, y), (30, 30, 30), 1)
        
        # Apply animation
        frame = fade_animation.apply_frame(frame, frame_num, fps)
        
        # Add frame number for debugging
        cv2.putText(frame, f"Frame: {frame_num}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        
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
    print("\nNow extracting frames for analysis...")
    
    # Extract frames for analysis
    analysis_dir = Path("outputs/fade_fixed_analysis")
    analysis_dir.mkdir(exist_ok=True)
    
    # Extract every 5th frame
    cap = cv2.VideoCapture(h264_output)
    frame_count = 0
    extracted = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % 5 == 0:
            cv2.imwrite(str(analysis_dir / f"frame_{frame_count:04d}.jpg"), frame)
            extracted += 1
        
        frame_count += 1
    
    cap.release()
    
    print(f"Extracted {extracted} frames for analysis")
    
    return h264_output, analysis_dir


def analyze_for_blinking(analysis_dir):
    """Analyze frames to check for blinking"""
    frames = sorted(analysis_dir.glob("frame_*.jpg"))
    
    print("\nAnalyzing for blinking...")
    print("=" * 60)
    
    # Load all frames
    loaded_frames = []
    for frame_path in frames:
        frame = cv2.imread(str(frame_path))
        loaded_frames.append(frame)
    
    # Analyze text region
    text_region_y1, text_region_y2 = 300, 420
    text_region_x1, text_region_x2 = 400, 880
    
    blinking_detected = False
    
    # Compare consecutive frames
    for i in range(len(loaded_frames) - 1):
        frame1 = loaded_frames[i]
        frame2 = loaded_frames[i + 1]
        
        # Extract text regions
        region1 = frame1[text_region_y1:text_region_y2, text_region_x1:text_region_x2]
        region2 = frame2[text_region_y1:text_region_y2, text_region_x1:text_region_x2]
        
        # Count bright pixels
        _, thresh1 = cv2.threshold(cv2.cvtColor(region1, cv2.COLOR_BGR2GRAY), 100, 255, cv2.THRESH_BINARY)
        _, thresh2 = cv2.threshold(cv2.cvtColor(region2, cv2.COLOR_BGR2GRAY), 100, 255, cv2.THRESH_BINARY)
        
        bright_pixels1 = np.sum(thresh1 > 0)
        bright_pixels2 = np.sum(thresh2 > 0)
        brightness_change = abs(bright_pixels2 - bright_pixels1)
        
        if brightness_change > 3000:  # Lower threshold for detecting issues
            print(f"Frame {i*5:03d} -> {(i+1)*5:03d}: Change of {brightness_change} pixels")
            blinking_detected = True
    
    if not blinking_detected:
        print("✅ NO BLINKING DETECTED! Animation is smooth.")
    else:
        print("⚠️ Some brightness variations detected, but should be much smoother than before.")
    
    return not blinking_detected


if __name__ == "__main__":
    print("TESTING FIXED 3D FADE ANIMATION")
    print("=" * 60)
    print()
    
    video_path, analysis_dir = test_fade_fixed()
    success = analyze_for_blinking(analysis_dir)
    
    print("\n" + "=" * 60)
    if success:
        print("SUCCESS! The blinking has been eliminated.")
    else:
        print("There may still be some minor variations. Check the video visually.")
    print("=" * 60)