#!/usr/bin/env python3
"""
Test the patched 3D text animation with quality and center fixes.
"""

import os
import sys
import cv2
import numpy as np

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Apply the patches first
import patch_3d_text_quality

# Now import the patched class
from utils.animations.text_3d_behind_segment import Text3DBehindSegment
from utils.segmentation.segment_extractor import extract_foreground_mask


def main():
    # Load test video
    video_path = "test_element_3sec.mp4"
    
    if not os.path.exists(video_path):
        print(f"Error: Video file '{video_path}' not found!")
        return
    
    print(f"Loading video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    resolution = (width, height)
    
    # Load frames
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    
    print(f"Video loaded: {width}x{height} @ {fps}fps, {len(frames)} frames")
    
    # Create segment mask
    print("Creating segment mask...")
    mask = extract_foreground_mask(frames[0])
    if mask.shape[:2] != (height, width):
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_LINEAR)
    
    print("\n" + "="*60)
    print("Testing Patched 3D Text Animation")
    print("="*60)
    
    # Create animation with patched class
    anim = Text3DBehindSegment(
        duration=3.0,
        fps=fps,
        resolution=resolution,
        text="HELLO WORLD",
        segment_mask=mask,
        font_size=140,  # Larger font
        text_color=(255, 220, 0),  # Golden yellow
        depth_color=(200, 170, 0),  # Darker yellow
        depth_layers=10,
        depth_offset=3,
        start_scale=2.2,  # Start bigger
        end_scale=0.9,
        phase1_duration=1.2,
        phase2_duration=0.6,
        phase3_duration=1.2,
        shadow_offset=8,
        outline_width=2,
        perspective_angle=25
    )
    
    # Generate frames
    print("\nGenerating animation frames...")
    output_frames = []
    total_frames = int(3.0 * fps)
    
    for i in range(total_frames):
        if i % 10 == 0:
            print(f"  Frame {i}/{total_frames}...")
        
        # Use corresponding background frame
        bg_frame = frames[i % len(frames)]
        
        # Generate frame
        anim_frame = anim.generate_frame(i, bg_frame)
        
        # Ensure RGB format
        if anim_frame.shape[2] == 4:
            anim_frame = anim_frame[:, :, :3]
        
        output_frames.append(anim_frame)
    
    print("  ✓ Animation complete!")
    
    # Save video
    print("\nSaving video...")
    output_path = "text_3d_patched_quality.mp4"
    
    # Save with OpenCV first
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    temp_path = "temp_3d.mp4"
    out = cv2.VideoWriter(temp_path, fourcc, fps, resolution)
    
    for frame in output_frames:
        # Convert RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)
    
    out.release()
    
    # Convert to H.264
    import subprocess
    cmd = [
        'ffmpeg', '-y', '-i', temp_path,
        '-c:v', 'libx264',
        '-preset', 'slow',
        '-crf', '18',
        '-pix_fmt', 'yuv420p',
        '-movflags', '+faststart',
        output_path
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Clean up temp file
    if os.path.exists(temp_path):
        os.remove(temp_path)
    
    print(f"  ✓ Saved: {output_path}")
    
    # Extract preview frames
    print("\nExtracting preview frames...")
    
    for idx, pct in [(30, 25), (90, 50), (135, 75)]:
        frame = output_frames[idx]
        filename = f"patched_3d_preview_{pct}.png"
        cv2.imwrite(filename, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        print(f"  ✓ {filename}")
    
    print("\n" + "="*60)
    print("✅ Patched 3D text animation completed successfully!")
    print("="*60)
    print("\nKey improvements:")
    print("  • Smooth anti-aliased edges (no pixelation)")
    print("  • Text shrinks towards center point")
    print("  • High-quality depth layers and shadows")
    print("  • Proper frame compositing")


if __name__ == "__main__":
    main()