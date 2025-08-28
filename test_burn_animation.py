#!/usr/bin/env python3
"""
Test the 3D burn animation effect.
"""

import cv2
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.animations.letter_3d_burn import Letter3DBurn


def create_burn_animation_demo():
    """Create a demo of the burn animation."""
    
    # Video properties
    width, height = 1280, 720
    fps = 30
    duration = 3.0
    
    # Create burn animation
    burn = Letter3DBurn(
        duration=duration,
        fps=fps,
        resolution=(width, height),
        text="BURN EFFECT",
        font_size=120,
        text_color=(255, 220, 0),  # Yellow text
        burn_color=(255, 50, 0),    # Orange burn
        initial_position=(width // 2, height // 2),
        stable_duration=0.2,
        burn_duration=0.8,
        burn_stagger=0.15,
        smoke_rise_distance=150,
        reverse_order=False,  # Burn from left to right
        is_behind=False,  # No occlusion for demo
        supersample_factor=4,  # High quality
        debug=True
    )
    
    # Create background gradient
    background = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height):
        gray_value = int(30 + (y / height) * 50)  # Gradient from dark to lighter
        background[y, :] = (gray_value, gray_value, gray_value + 10)
    
    # Generate frames
    output_path = "outputs/burn_effect_demo.mp4"
    os.makedirs("outputs", exist_ok=True)
    
    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"Generating {int(duration * fps)} frames of burn animation...")
    
    for frame_num in range(int(duration * fps)):
        # Generate frame
        frame = burn.generate_frame(frame_num, background)
        
        # Write frame
        writer.write(frame)
        
        # Progress
        if frame_num % 10 == 0:
            progress = (frame_num / (duration * fps)) * 100
            print(f"Progress: {progress:.1f}%")
    
    writer.release()
    print(f"✅ Burn animation saved to: {output_path}")
    
    # Convert to H.264 for compatibility
    h264_output = "outputs/burn_effect_demo_h264.mp4"
    cmd = f"ffmpeg -i {output_path} -c:v libx264 -preset fast -crf 23 -pix_fmt yuv420p -movflags +faststart -y {h264_output}"
    os.system(cmd)
    print(f"✅ H.264 version saved to: {h264_output}")
    
    return h264_output


def compare_burn_vs_dissolve():
    """Create side-by-side comparison of burn and dissolve effects."""
    
    from utils.animations.letter_3d_dissolve import Letter3DDissolve
    
    # Common parameters
    width, height = 640, 360
    fps = 30
    duration = 3.0
    text = "HELLO"
    font_size = 80
    
    # Create animations
    burn = Letter3DBurn(
        duration=duration,
        fps=fps,
        resolution=(width, height),
        text=text,
        font_size=font_size,
        text_color=(255, 220, 0),
        initial_position=(width // 2, height // 3),
        burn_stagger=0.1,
        supersample_factor=2
    )
    
    dissolve = Letter3DDissolve(
        duration=duration,
        fps=fps,
        resolution=(width, height),
        text=text,
        font_size=font_size,
        text_color=(255, 220, 0),
        initial_position=(width // 2, 2 * height // 3),
        dissolve_stagger=0.1,
        supersample_factor=2
    )
    
    # Create output video
    output_path = "outputs/burn_vs_dissolve_comparison.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print("Creating burn vs dissolve comparison...")
    
    # Background
    background = np.ones((height, width, 3), dtype=np.uint8) * 40
    
    for frame_num in range(int(duration * fps)):
        frame = background.copy()
        
        # Add labels
        cv2.putText(frame, "BURN", (50, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        cv2.putText(frame, "DISSOLVE", (50, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        
        # Generate animations
        frame = burn.generate_frame(frame_num, frame)
        frame = dissolve.generate_frame(frame_num, frame)
        
        writer.write(frame)
    
    writer.release()
    
    # Convert to H.264
    h264_output = "outputs/burn_vs_dissolve_comparison_h264.mp4"
    cmd = f"ffmpeg -i {output_path} -c:v libx264 -preset fast -crf 23 -pix_fmt yuv420p -movflags +faststart -y {h264_output}"
    os.system(cmd)
    
    print(f"✅ Comparison saved to: {h264_output}")
    return h264_output


if __name__ == "__main__":
    print("="*60)
    print("3D BURN ANIMATION TEST")
    print("="*60)
    
    # Create burn demo
    burn_video = create_burn_animation_demo()
    
    # Create comparison
    comparison_video = compare_burn_vs_dissolve()
    
    print("\n" + "="*60)
    print("RESULTS:")
    print("="*60)
    print(f"✅ Burn effect demo: {burn_video}")
    print(f"✅ Comparison video: {comparison_video}")
    print("\n💡 Features demonstrated:")
    print("  • Letters burn from edges inward")
    print("  • Fire particles at burning edges")
    print("  • Smoke particles rising upward")
    print("  • Gradual charring effect")
    print("  • Staggered burn timing")
    print("="*60)