"""
Debug and fix the flashing issue in 3D Fade Wave animation
Focus on first 3 seconds only
"""

import cv2
import numpy as np
import sys
import os

# Add animation modules to path
sys.path.append('utils/animations')
sys.path.append('utils/animations/3d_animations')
sys.path.append('utils/animations/3d_animations/opacity_3d')

from base_3d_text_animation import Animation3DConfig, Letter3D
from opacity_3d import Fade3D


class DebugFade3D(Fade3D):
    """Fade3D with extensive debugging"""
    
    def update_letters(self, progress: float, frame_number: int, fps: float):
        """Override to add debugging"""
        # Call parent implementation
        super().update_letters(progress, frame_number, fps)
        
        # Debug every 5 frames
        if frame_number % 5 == 0:
            print(f"\nFrame {frame_number} (progress: {progress:.3f}):")
            for i, letter in enumerate(self.letters[:3]):  # Just first 3 letters
                print(f"  Letter '{letter.character}': opacity={letter.opacity:.3f}")
    
    def _create_letter_sprite(self, letter: Letter3D):
        """Debug sprite creation"""
        super()._create_letter_sprite(letter)
        if letter.index == 0:  # Just debug first letter
            sprite = letter._rendered_sprite
            if sprite is not None:
                alpha = sprite[:, :, 3]
                print(f"  Sprite created for '{letter.character}': "
                      f"shape={sprite.shape}, max_alpha={np.max(alpha)}, "
                      f"mean_alpha={np.mean(alpha[alpha > 0]):.1f}")


def test_fade_first_3_seconds():
    """Test just the first 3 seconds with debugging"""
    
    print("Testing 3D Fade Wave - First 3 seconds only")
    print("=" * 60)
    
    # Load ai_math1.mp4
    input_video = "uploads/assets/videos/ai_math1.mp4"
    cap = cv2.VideoCapture(input_video)
    
    if not cap.isOpened():
        print(f"Error: Could not open {input_video}")
        return None
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video: {width}x{height} @ {fps:.2f} fps")
    
    # Create output for just 3 seconds
    output_path = "outputs/fade_debug_3sec.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Create the Fade3D animation
    config = Animation3DConfig(
        text="3D FADE WAVE",
        duration_ms=3000,  # Exactly 3 seconds
        position=(640, 360, 0),
        font_size=70,
        font_color=(255, 255, 255),  # Pure white
        depth_color=(100, 100, 100),
        stagger_ms=40
    )
    
    print(f"\nAnimation config:")
    print(f"  Duration: {config.duration_ms}ms")
    print(f"  Font color: {config.font_color}")
    print(f"  Stagger: {config.stagger_ms}ms")
    
    # Test different configurations
    test_configs = [
        # Test 1: Original with wave mode
        {"fade_mode": "wave", "depth_fade": False, "name": "Wave (Original)"},
        # Test 2: Uniform mode (no wave)
        {"fade_mode": "uniform", "depth_fade": False, "name": "Uniform (No Wave)"},
        # Test 3: Try without stagger
        {"fade_mode": "uniform", "depth_fade": False, "name": "No Stagger", "stagger": 0},
    ]
    
    for test_idx, test_config in enumerate(test_configs):
        print(f"\n\nTEST {test_idx + 1}: {test_config['name']}")
        print("-" * 40)
        
        # Adjust config if needed
        if "stagger" in test_config:
            config.stagger_ms = test_config["stagger"]
        else:
            config.stagger_ms = 40
        
        # Create animation
        fade_anim = DebugFade3D(
            config, 
            fade_mode=test_config["fade_mode"],
            depth_fade=test_config["depth_fade"],
            start_opacity=0.0,
            end_opacity=1.0
        )
        
        # Reset video to start
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        # Process 3 seconds of frames
        total_frames = int(3 * fps)
        
        print(f"Processing {total_frames} frames...")
        
        for frame_num in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                print(f"Failed to read frame {frame_num}")
                break
            
            # Apply animation
            animated_frame = fade_anim.apply_frame(frame, frame_num, fps)
            
            # Add debug info
            cv2.putText(animated_frame, f"Test {test_idx + 1}: {test_config['name']}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
            cv2.putText(animated_frame, f"Frame: {frame_num}/{total_frames}", 
                       (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            out.write(animated_frame)
            
            # Analyze for flashing
            if frame_num > 0 and frame_num % 5 == 0:
                # Compare brightness in text region
                text_region = animated_frame[320:400, 400:880]
                gray = cv2.cvtColor(text_region, cv2.COLOR_BGR2GRAY)
                mean_brightness = np.mean(gray)
                
                # Store for comparison
                if not hasattr(test_fade_first_3_seconds, 'prev_brightness'):
                    test_fade_first_3_seconds.prev_brightness = mean_brightness
                else:
                    brightness_change = abs(mean_brightness - test_fade_first_3_seconds.prev_brightness)
                    if brightness_change > 20:  # Significant change
                        print(f"  ⚠️ Frame {frame_num}: Large brightness change: {brightness_change:.1f}")
                    test_fade_first_3_seconds.prev_brightness = mean_brightness
        
        # Only process first config for now
        if test_idx == 0:
            break
    
    out.release()
    cap.release()
    
    # Convert to H.264
    h264_output = output_path.replace('.mp4', '_h264.mp4')
    convert_cmd = f"ffmpeg -i {output_path} -c:v libx264 -preset fast -crf 23 -pix_fmt yuv420p -movflags +faststart {h264_output} -y"
    os.system(convert_cmd)
    os.remove(output_path)
    
    print(f"\n✅ Debug video created: {h264_output}")
    return h264_output


def analyze_flashing_cause():
    """Analyze what might be causing the flashing"""
    
    print("\n" + "=" * 60)
    print("ANALYZING POTENTIAL CAUSES OF FLASHING")
    print("=" * 60)
    
    print("\nPossible causes:")
    print("1. Sprite caching issue - sprites might be recreated each frame")
    print("2. Wave offset calculation causing rapid changes")
    print("3. Stagger timing causing letters to appear/disappear")
    print("4. Alpha blending issue in compositing")
    
    # Check if sprites are being cached properly
    from base_3d_text_animation import Base3DTextAnimation, Letter3D
    
    # Create a test letter
    test_letter = Letter3D(
        character='A',
        index=0,
        position=np.array([100, 100, 0], dtype=np.float32),
        rotation=np.zeros(3, dtype=np.float32),
        scale=np.ones(3, dtype=np.float32),
        opacity=1.0,
        color=(255, 255, 255),
        depth_color=(100, 100, 100)
    )
    
    print("\nChecking sprite caching:")
    print(f"  Initial sprite: {test_letter._rendered_sprite}")
    
    # The issue might be that sprites are recreated when opacity changes
    # Let's verify the caching behavior


if __name__ == "__main__":
    print("3D FADE WAVE FLASHING DEBUG")
    print("=" * 60)
    
    # First analyze potential causes
    analyze_flashing_cause()
    
    # Then test with debug output
    output = test_fade_first_3_seconds()
    
    if output:
        print("\n" + "=" * 60)
        print("DEBUG COMPLETE")
        print("Check the video for flashing")
        print("Look at the debug output above for opacity changes")
        print("=" * 60)