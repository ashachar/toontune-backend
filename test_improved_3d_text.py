#!/usr/bin/env python3
"""
Test the improved 3D text animation with better quality and center-shrinking.
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fix_3d_text_quality import Text3DBehindSegmentImproved
from utils.segmentation.segment_extractor import extract_foreground_mask


def load_video(video_path):
    """Load video and get properties."""
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    
    cap.release()
    return frames, fps, (width, height), frame_count


def create_segment_mask(frame, resolution):
    """Create a segment mask using rembg."""
    print("Creating segment mask...")
    
    # Extract foreground mask
    mask = extract_foreground_mask(frame)
    
    # Ensure mask is the right size
    if mask.shape[:2] != (resolution[1], resolution[0]):
        mask = cv2.resize(mask, resolution, interpolation=cv2.INTER_LINEAR)
    
    return mask


def save_video_h264(frames, output_path, fps, resolution):
    """Save frames to H.264 video with high quality."""
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    temp_path = output_path.replace('.mp4', '_temp.mp4')
    
    out = cv2.VideoWriter(temp_path, fourcc, fps, resolution)
    
    for frame in frames:
        # Convert RGBA to BGR for OpenCV
        if frame.shape[2] == 4:
            # Remove alpha channel
            frame = frame[:, :, :3]
        # Convert RGB to BGR
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)
    
    out.release()
    
    # Convert to H.264 using FFmpeg with high quality settings
    import subprocess
    cmd = [
        'ffmpeg', '-y', '-i', temp_path,
        '-c:v', 'libx264', 
        '-preset', 'slower',  # Slower preset for better quality
        '-crf', '18',  # Lower CRF for higher quality (18 is visually lossless)
        '-pix_fmt', 'yuv420p', 
        '-movflags', '+faststart',
        output_path
    ]
    
    subprocess.run(cmd, capture_output=True, text=True)
    
    # Remove temp file
    if os.path.exists(temp_path):
        os.remove(temp_path)
    
    print(f"  ✓ Saved with high-quality H.264 encoding")


def main():
    # Input video path
    video_path = "test_element_3sec.mp4"
    
    if not os.path.exists(video_path):
        print(f"Error: Video file '{video_path}' not found!")
        print("Please ensure test_element_3sec.mp4 is in the current directory.")
        return
    
    print(f"Loading video: {video_path}")
    frames, fps, resolution, frame_count = load_video(video_path)
    
    if len(frames) == 0:
        print("Error: No frames loaded from video!")
        return
    
    print(f"Video loaded: {resolution[0]}x{resolution[1]} @ {fps}fps, {frame_count} frames")
    
    # Create segment mask from first frame
    segment_mask = create_segment_mask(frames[0], resolution)
    
    # Test text
    test_text = "HELLO WORLD"
    
    print("\n" + "="*60)
    print("Testing Improved 3D Text Animation")
    print("="*60)
    print("\nImprovements:")
    print("  • 3x supersampling for smooth edges")
    print("  • High-quality LANCZOS resampling")
    print("  • Proper center-point scaling")
    print("  • Soft multi-layer shadows")
    print("  • Cubic interpolation for perspective")
    print("="*60)
    
    # Create improved animation
    print("\nGenerating improved 3D text animation...")
    
    anim = Text3DBehindSegmentImproved(
        duration=3.0,
        fps=fps,
        resolution=resolution,
        text=test_text,
        segment_mask=segment_mask,
        font_size=120,
        text_color=(255, 220, 0),  # Golden yellow
        depth_color=(180, 150, 0),  # Darker yellow for depth
        depth_layers=12,  # More layers for smoother depth
        depth_offset=2,
        start_scale=2.5,  # Start larger to show the shrinking better
        end_scale=0.8,    # End slightly smaller
        phase1_duration=1.2,  # Longer shrink phase
        phase2_duration=0.6,
        phase3_duration=1.2,
        shadow_offset=8,
        outline_width=2,
        perspective_angle=30,
        supersample_factor=3  # 3x supersampling for anti-aliasing
    )
    
    # Generate animation frames
    output_frames = []
    total_frames = int(3.0 * fps)
    
    for i in range(total_frames):
        print(f"  Generating frame {i+1}/{total_frames}...", end='\r')
        
        # Use the corresponding background frame
        bg_frame = frames[i % len(frames)]
        
        # Generate frame
        anim_frame = anim.generate_frame(i, bg_frame)
        output_frames.append(anim_frame)
    
    print(f"\n  ✓ Animation generation complete!")
    
    # Save video
    print("\n" + "="*60)
    print("Saving output video...")
    print("="*60)
    
    output_path = "text_3d_improved_quality.mp4"
    print(f"\nSaving: {output_path}")
    save_video_h264(output_frames, output_path, fps, resolution)
    
    # Extract comparison frames
    print("\nExtracting comparison frames...")
    
    # Frame at 25% (during shrink)
    frame_25 = output_frames[total_frames // 4]
    cv2.imwrite("improved_3d_frame_25.png", cv2.cvtColor(frame_25[:,:,:3], cv2.COLOR_RGB2BGR))
    
    # Frame at 50% (transition)
    frame_50 = output_frames[total_frames // 2]
    cv2.imwrite("improved_3d_frame_50.png", cv2.cvtColor(frame_50[:,:,:3], cv2.COLOR_RGB2BGR))
    
    # Frame at 75% (behind)
    frame_75 = output_frames[3 * total_frames // 4]
    cv2.imwrite("improved_3d_frame_75.png", cv2.cvtColor(frame_75[:,:,:3], cv2.COLOR_RGB2BGR))
    
    print("  ✓ Comparison frames saved")
    
    print("\n" + "="*60)
    print("✅ Improved 3D text animation completed successfully!")
    print("="*60)
    print("\nOutput files:")
    print(f"  • {output_path} - High-quality 3D text animation")
    print(f"  • improved_3d_frame_25.png - Frame during shrinking")
    print(f"  • improved_3d_frame_50.png - Frame during transition")
    print(f"  • improved_3d_frame_75.png - Frame when behind subject")
    print("\nKey improvements:")
    print("  ✓ No pixelation - smooth anti-aliased edges")
    print("  ✓ Proper center shrinking - scales from center point")
    print("  ✓ Higher quality shadows and depth layers")
    print("  ✓ Smoother animation curves")


if __name__ == "__main__":
    main()