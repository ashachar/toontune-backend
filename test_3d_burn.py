#!/usr/bin/env python3
"""Test 3D burn animation with photorealistic effects."""

import cv2
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.animations.letter_3d_burn.burn import Letter3DBurn

def main():
    print("Creating 3D burn animation...")
    
    # Animation parameters
    width, height = 1280, 720
    fps = 30
    duration = 3.5
    
    # Create burn animation with 3D letters
    burn = Letter3DBurn(
        duration=duration,
        fps=fps,
        resolution=(width, height),
        text="FIRE",
        font_size=150,
        text_color=(255, 220, 0),      # Golden yellow
        burn_color=(255, 100, 0),       # Orange fire
        depth_layers=10,                # More layers for pronounced 3D
        depth_offset=4,                  # Bigger offset for depth
        burn_duration=0.8,
        burn_stagger=0.2,
        supersample_factor=2,
        debug=False
    )
    
    # Create dark background with slight gradient
    background = np.zeros((height, width, 3), dtype=np.uint8)
    # Add subtle gradient
    for y in range(height):
        background[y, :] = int(40 * (1 - y / height))
    
    # Setup video writer
    output_path = "outputs/letter_3d_burn_demo.mp4"
    os.makedirs("outputs", exist_ok=True)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    total_frames = int(duration * fps)
    print(f"Rendering {total_frames} frames...")
    
    # Render frames
    for frame_num in range(total_frames):
        frame = burn.generate_frame(frame_num, background.copy())
        writer.write(frame)
        
        if frame_num % 30 == 0:
            print(f"  Frame {frame_num}/{total_frames}")
    
    writer.release()
    
    # Convert to H.264
    h264_output = "outputs/letter_3d_burn_demo_h264.mp4"
    cmd = f"ffmpeg -i {output_path} -c:v libx264 -preset fast -crf 23 -pix_fmt yuv420p -movflags +faststart -y {h264_output} 2>/dev/null"
    os.system(cmd)
    
    print(f"\n✅ 3D burn animation saved to: {h264_output}")
    
    # Also create a version with different text
    print("\nCreating second demo with 'BURN' text...")
    
    burn2 = Letter3DBurn(
        duration=duration,
        fps=fps,
        resolution=(width, height),
        text="BURN",
        font_size=180,
        text_color=(255, 255, 200),     # Almost white hot
        burn_color=(255, 200, 100),     # Very hot fire
        depth_layers=12,
        depth_offset=5,
        burn_duration=0.6,
        burn_stagger=0.15,
        supersample_factor=2
    )
    
    output_path2 = "outputs/letter_3d_burn_demo2.mp4"
    writer2 = cv2.VideoWriter(output_path2, fourcc, fps, (width, height))
    
    for frame_num in range(total_frames):
        frame = burn2.generate_frame(frame_num, background.copy())
        writer2.write(frame)
    
    writer2.release()
    
    # Convert to H.264
    h264_output2 = "outputs/letter_3d_burn_demo2_h264.mp4"
    os.system(f"ffmpeg -i {output_path2} -c:v libx264 -preset fast -crf 23 -pix_fmt yuv420p -movflags +faststart -y {h264_output2} 2>/dev/null")
    
    print(f"✅ Second 3D burn animation saved to: {h264_output2}")

if __name__ == "__main__":
    main()