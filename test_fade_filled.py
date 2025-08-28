"""
Test 3D Fade Wave animation with properly filled text
"""

import cv2
import numpy as np
import sys
import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

# Add animation modules to path
sys.path.append('utils/animations')
sys.path.append('utils/animations/3d_animations')
sys.path.append('utils/animations/3d_animations/opacity_3d')

from base_3d_text_animation import Animation3DConfig
from opacity_3d import Fade3D


def test_fade_filled():
    """Test 3D Fade Wave with filled text"""
    
    print("Testing 3D Fade Wave with filled text...")
    print("=" * 60)
    
    # First, let's verify how PIL renders text
    print("\nTesting PIL text rendering...")
    test_img = Image.new('RGBA', (400, 200), (0, 0, 0, 255))
    test_draw = ImageDraw.Draw(test_img)
    
    # Try different font options
    try:
        # Try Arial Black or other bold font for better fill
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 80, index=1)  # Try bold variant
    except:
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 80)
        except:
            font = ImageFont.load_default()
    
    # Draw test text
    test_draw.text((200, 100), "TEST", fill=(255, 255, 255, 255), font=font, anchor='mm')
    test_img.save("outputs/pil_text_test.png")
    print("PIL test saved to: outputs/pil_text_test.png")
    
    # Create video
    width, height = 1280, 720
    fps = 30
    duration_seconds = 5
    total_frames = fps * duration_seconds
    
    # Create output video
    output_path = "outputs/test_fade_filled.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Create 3D Fade animation with bright white text
    config = Animation3DConfig(
        text="3D FADE WAVE",
        duration_ms=4000,
        position=(640, 360, 0),
        font_size=100,  # Larger size
        font_color=(255, 255, 255),  # Pure bright white
        depth_color=(100, 100, 100),  # Darker shadow for contrast
        depth_layers=8,
        stagger_ms=50
    )
    
    # Try specifying a bold font if available
    try:
        # Try to use Arial Bold or Helvetica Bold
        config.font_path = "/System/Library/Fonts/HelveticaNeue.ttc"  # Try HelveticaNeue which might have better fill
    except:
        pass
    
    fade_animation = Fade3D(config, fade_mode="wave")
    
    # Render frames
    for frame_num in range(total_frames):
        # Create black background
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Add subtle grid
        for x in range(0, width, 100):
            cv2.line(frame, (x, 0), (x, height), (30, 30, 30), 1)
        for y in range(0, height, 100):
            cv2.line(frame, (0, y), (width, y), (30, 30, 30), 1)
        
        # Apply animation
        frame = fade_animation.apply_frame(frame, frame_num, fps)
        
        # Add info text
        cv2.putText(frame, "3D Fade Wave - Testing Filled Text", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
        cv2.putText(frame, f"Frame: {frame_num}/{total_frames}", 
                   (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
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
    
    # Extract a frame to examine the text
    cap = cv2.VideoCapture(h264_output)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 60)  # Frame at 2 seconds
    ret, frame = cap.read()
    if ret:
        cv2.imwrite("outputs/fade_text_frame.jpg", frame)
        print("Sample frame saved: outputs/fade_text_frame.jpg")
    cap.release()
    
    return h264_output


if __name__ == "__main__":
    print("3D FADE WAVE - FILLED TEXT TEST")
    print("=" * 60)
    print()
    
    output = test_fade_filled()
    
    print("\n" + "=" * 60)
    print("COMPLETE!")
    print("Check the output to see if text is properly filled")
    print("=" * 60)