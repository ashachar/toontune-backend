"""
Debug script to trace opacity issues in 3D Fade animation
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


class DebugFade3D(Fade3D):
    """Fade3D with debug output"""
    
    def update_letters(self, progress: float, frame_number: int, fps: float):
        """Override to add debug output"""
        super().update_letters(progress, frame_number, fps)
        
        # Debug output for last frame
        if frame_number >= 119:  # Near end of 4-second animation at 30fps
            print(f"\n=== FRAME {frame_number} (progress: {progress:.2f}) ===")
            for i, letter in enumerate(self.letters):
                print(f"Letter '{letter.character}' [{i}]:")
                print(f"  - opacity: {letter.opacity:.3f}")
                print(f"  - color: {letter.color}")
                print(f"  - position Z: {letter.position[2]:.1f}")


def debug_sprite_creation():
    """Debug the sprite creation process"""
    print("\n=== DEBUGGING SPRITE CREATION ===")
    
    # Create a test letter
    from base_3d_text_animation import Letter3D
    test_letter = Letter3D(
        character='A',
        index=0,
        position=np.array([100, 100, 0], dtype=np.float32),
        rotation=np.zeros(3, dtype=np.float32),
        scale=np.ones(3, dtype=np.float32),
        opacity=1.0,  # FULL OPACITY
        color=(255, 255, 255),  # WHITE
        depth_color=(100, 100, 100)
    )
    
    # Create sprite manually
    img = Image.new('RGBA', (200, 200), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 80)
    except:
        font = ImageFont.load_default()
    
    # Draw shadow
    shadow_color = (60, 60, 60, 255)  # FULLY OPAQUE SHADOW
    print(f"Shadow color: {shadow_color}")
    draw.text((110, 102), 'A', fill=shadow_color, font=font, anchor='mm', stroke_width=1, stroke_fill=shadow_color)
    
    # Draw main letter
    main_color = (255, 255, 255, 255)  # FULLY OPAQUE WHITE
    print(f"Main color: {main_color}")
    draw.text((100, 100), 'A', fill=main_color, font=font, anchor='mm', stroke_width=2, stroke_fill=main_color)
    
    # Check the result
    sprite = np.array(img)
    
    # Find non-transparent pixels
    alpha = sprite[:, :, 3]
    non_zero = np.where(alpha > 0)
    
    if len(non_zero[0]) > 0:
        max_alpha = np.max(alpha)
        mean_alpha = np.mean(alpha[alpha > 0])
        
        # Sample some pixels from the center
        center_y, center_x = 100, 100
        sample_region = alpha[center_y-10:center_y+10, center_x-10:center_x+10]
        sample_max = np.max(sample_region)
        
        print(f"\nSprite analysis:")
        print(f"  - Non-zero pixels: {len(non_zero[0])}")
        print(f"  - Max alpha: {max_alpha}/255")
        print(f"  - Mean alpha (non-zero): {mean_alpha:.1f}/255")
        print(f"  - Center region max alpha: {sample_max}/255")
        
        # Check actual RGB values where alpha is high
        high_alpha_mask = alpha > 200
        if np.any(high_alpha_mask):
            rgb_values = sprite[high_alpha_mask][:5]  # Sample first 5 pixels
            print(f"  - Sample RGB values (high alpha): {rgb_values[:, :3]}")
    
    img.save("outputs/debug_letter_sprite.png")
    print(f"\nDebug sprite saved to: outputs/debug_letter_sprite.png")


def test_fade_debug():
    """Test with extensive debugging"""
    
    print("Testing 3D Fade with opacity debugging...")
    print("=" * 60)
    
    # First debug sprite creation
    debug_sprite_creation()
    
    # Create video
    width, height = 1280, 720
    fps = 30
    duration_seconds = 4
    total_frames = fps * duration_seconds
    
    # Create output video
    output_path = "outputs/debug_fade.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Create debug fade animation
    config = Animation3DConfig(
        text="TEST",
        duration_ms=3500,  # Slightly shorter to ensure it completes
        position=(640, 360, 0),
        font_size=120,  # Even larger
        font_color=(255, 255, 255),  # Pure white
        depth_color=(100, 100, 100),
        depth_layers=8,
        stagger_ms=0  # No stagger for cleaner debug
    )
    
    fade_animation = DebugFade3D(config, 
                                 start_opacity=0.0,
                                 end_opacity=1.0,
                                 fade_mode="uniform",  # Simple mode
                                 depth_fade=False)  # Disable depth fade
    
    print(f"\nAnimation config:")
    print(f"  - Duration: {config.duration_ms}ms")
    print(f"  - Start opacity: 0.0")
    print(f"  - End opacity: 1.0")
    print(f"  - Depth fade: DISABLED")
    print(f"  - Fade mode: uniform")
    
    # Render frames
    for frame_num in range(total_frames):
        # Create BLACK background (not transparent)
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Apply animation
        frame = fade_animation.apply_frame(frame, frame_num, fps)
        
        # Add debug info
        progress = frame_num / (fps * (config.duration_ms / 1000))
        cv2.putText(frame, f"Frame: {frame_num} Progress: {progress:.2f}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
        
        out.write(frame)
        
        if frame_num == total_frames - 1:
            # Save last frame for inspection
            cv2.imwrite("outputs/debug_last_frame.jpg", frame)
            print(f"\nLast frame saved to: outputs/debug_last_frame.jpg")
            
            # Analyze the last frame
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            text_region = gray[300:420, 540:740]  # Center region where text should be
            max_brightness = np.max(text_region)
            mean_brightness = np.mean(text_region)
            print(f"\nLast frame analysis:")
            print(f"  - Max brightness in text region: {max_brightness}/255")
            print(f"  - Mean brightness in text region: {mean_brightness:.1f}/255")
    
    out.release()
    
    # Convert to H.264
    h264_output = output_path.replace('.mp4', '_h264.mp4')
    convert_cmd = f"ffmpeg -i {output_path} -c:v libx264 -preset fast -crf 23 -pix_fmt yuv420p -movflags +faststart {h264_output} -y"
    os.system(convert_cmd)
    os.remove(output_path)
    
    print(f"\n✅ Debug video created: {h264_output}")
    
    return h264_output


if __name__ == "__main__":
    print("OPACITY DEBUG ANALYSIS")
    print("=" * 60)
    print()
    
    output = test_fade_debug()
    
    print("\n" + "=" * 60)
    print("DEBUG COMPLETE!")
    print("Check outputs/debug_last_frame.jpg to see the final frame")
    print("=" * 60)