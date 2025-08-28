"""
Test the new Digital/Glitch 3D animations
"""

import cv2
import numpy as np
import sys
import os

# Add animation modules to path
sys.path.append('utils/animations')
sys.path.append('utils/animations/3d_animations')
sys.path.append('utils/animations/3d_animations/digital_3d')

from base_3d_text_animation import Animation3DConfig
from digital_3d import Glitch3D, Digital3D, Hologram3D, Static3D


def test_digital_animations():
    """Test all digital/tech 3D animations"""
    
    print("Testing Digital/Glitch 3D Animations")
    print("=" * 60)
    
    # Load input video
    input_video = "uploads/assets/videos/ai_math1.mp4"
    cap = cv2.VideoCapture(input_video)
    
    if not cap.isOpened():
        print(f"Error: Could not open {input_video}")
        return None
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video: {width}x{height} @ {fps:.2f} fps")
    
    # Create output video
    output_path = "outputs/digital_3d_showcase.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Define animations to test
    animations = [
        ("GLITCH 3D", Glitch3D, {
            "glitch_intensity": 0.7,
            "glitch_frequency": 0.4,
            "rgb_shift_amount": 15,
            "scan_lines": True,
            "digital_noise": True,
            "displacement": True
        }, 3500),
        
        ("DIGITAL MATRIX", Digital3D, {
            "matrix_rain": True,
            "binary_fade": True,
            "terminal_cursor": True
        }, 3500),
        
        ("HOLOGRAM 3D", Hologram3D, {
            "scan_speed": 2.0,
            "flicker_amount": 0.3,
            "chromatic_aberration": True
        }, 3500),
        
        ("TV STATIC", Static3D, {
            "static_intensity": 0.8,
            "tune_in_effect": True,
            "horizontal_hold": True
        }, 3500),
    ]
    
    print(f"\nTotal animations to render: {len(animations)}")
    
    frame_count = 0
    animation_index = 0
    current_animation = None
    animation_start_frame = 0
    
    # Process frames
    while animation_index <= len(animations):
        ret, frame = cap.read()
        if not ret:
            # Loop video if needed
            if animation_index < len(animations):
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
                if not ret:
                    break
            else:
                break
        
        # Check if we need to start a new animation
        if animation_index < len(animations) and (current_animation is None or (frame_count - animation_start_frame) * 1000 / fps >= animations[animation_index][3]):
            # Create new animation
            text, anim_class, params, duration = animations[animation_index]
            print(f"\n[{animation_index + 1}/{len(animations)}] Starting: {text}")
            
            config = Animation3DConfig(
                text=text,
                duration_ms=duration,
                position=(640, 360, 0),
                font_size=80,
                font_color=(255, 255, 255),
                depth_color=(100, 100, 100),
                stagger_ms=40,
                enable_glow=True,
                glow_radius=8
            )
            
            current_animation = anim_class(config, **params)
            animation_start_frame = frame_count
            animation_index += 1
        
        # Apply current animation
        if current_animation and animation_index > 0:
            animation_frame_num = frame_count - animation_start_frame
            animated_frame = current_animation.apply_frame(frame, animation_frame_num, fps)
            
            # Apply post-processing effects if available
            if hasattr(current_animation, 'apply_frame_post_effects'):
                progress = min(1.0, (animation_frame_num / fps) * 1000 / animations[animation_index-1][3])
                animated_frame = current_animation.apply_frame_post_effects(animated_frame, progress)
            
            # Add label
            cv2.putText(animated_frame, f"{animation_index}/{len(animations)}: {animations[animation_index-1][0]}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
            cv2.putText(animated_frame, "Digital/Glitch Animation Test", 
                       (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
            
            out.write(animated_frame)
        else:
            out.write(frame)
        
        frame_count += 1
        
        if frame_count % 50 == 0:
            print(f"  Progress: {frame_count} frames")
    
    out.release()
    cap.release()
    
    # Convert to H.264
    print(f"\nConverting to H.264...")
    h264_output = output_path.replace('.mp4', '_h264.mp4')
    convert_cmd = f"ffmpeg -i {output_path} -c:v libx264 -preset fast -crf 23 -pix_fmt yuv420p -movflags +faststart {h264_output} -y"
    os.system(convert_cmd)
    os.remove(output_path)
    
    print(f"\n" + "=" * 60)
    print(f"DIGITAL ANIMATION TEST COMPLETE!")
    print(f"Total animations: {len(animations)}")
    print(f"Total duration: {frame_count / fps:.1f} seconds")
    print(f"Output: {h264_output}")
    print("=" * 60)
    
    return h264_output


if __name__ == "__main__":
    print("DIGITAL/GLITCH 3D ANIMATION TEST")
    print("=" * 60)
    print("Testing animations that might appear at 9-11s in real_estate.mp4:")
    print("  • Glitch3D - Digital interference with RGB shifts")
    print("  • Digital3D - Matrix-style digital rain")
    print("  • Hologram3D - Holographic projection effect")
    print("  • Static3D - TV static/noise reveal")
    print()
    
    output = test_digital_animations()
    
    if output:
        print(f"\n✅ Success! Video created: {output}")
        print("\nThese animations feature:")
        print("  • Digital glitch effects with RGB channel separation")
        print("  • Scan lines and digital noise")
        print("  • Random displacement and color corruption")
        print("  • Matrix rain and binary code reveals")
        print("  • Holographic scanning effects")
        print("  • TV static interference patterns")