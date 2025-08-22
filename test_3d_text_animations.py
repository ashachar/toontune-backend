#!/usr/bin/env python3
"""
Test script for 3D text animations.
Demonstrates Text3DBehindSegment and Word3DDissolve animations.
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.animations.text_3d_behind_segment import Text3DBehindSegment
from utils.animations.word_3d_dissolve import Word3DDissolve
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
    print("Testing 3D Text Animations")
    print("="*60)
    
    # Test 1: Text3DBehindSegment
    print("\n1. Testing Text3DBehindSegment...")
    
    anim1 = Text3DBehindSegment(
        duration=3.0,
        fps=fps,
        resolution=resolution,
        text=test_text,
        segment_mask=segment_mask,
        font_size=120,
        text_color=(255, 220, 0),  # Golden yellow
        depth_color=(180, 150, 0),  # Darker yellow for depth
        depth_layers=10,
        depth_offset=3,
        start_scale=2.0,
        end_scale=1.0,
        phase1_duration=1.0,
        phase2_duration=0.67,
        phase3_duration=1.33,
        shadow_offset=6,
        outline_width=3,
        perspective_angle=30
    )
    
    # Generate animation frames
    output_frames_1 = []
    for i, bg_frame in enumerate(frames[:int(3.0 * fps)]):
        print(f"  Generating frame {i+1}/{int(3.0 * fps)}...", end='\r')
        anim_frame = anim1.generate_frame(i, bg_frame)
        output_frames_1.append(anim_frame)
    print("\n  Text3DBehindSegment animation complete!")
    
    # Test 2: Word3DDissolve
    print("\n2. Testing Word3DDissolve...")
    
    anim2 = Word3DDissolve(
        duration=3.0,
        fps=fps,
        resolution=resolution,
        text=test_text,
        segment_mask=segment_mask,
        font_size=100,
        text_color=(255, 220, 0),
        depth_color=(180, 150, 0),
        depth_layers=8,
        depth_offset=2,
        dissolve_mode='letter',
        dissolve_duration=0.5,
        dissolve_overlap=0.3,
        particle_size=3,
        particle_velocity=(0, -3),
        particle_acceleration=(0, -0.15),
        particle_spread=3.0,
        fade_start=0.3,
        stable_duration=0.5,
        random_order=True,
        depth_dissolve_delay=0.03,
        perspective_angle=25
    )
    
    # Generate animation frames
    output_frames_2 = []
    for i, bg_frame in enumerate(frames[:int(3.0 * fps)]):
        print(f"  Generating frame {i+1}/{int(3.0 * fps)}...", end='\r')
        anim_frame = anim2.generate_frame(i, bg_frame)
        output_frames_2.append(anim_frame)
    print("\n  Word3DDissolve animation complete!")
    
    # Test 3: Combined animation (Text3DBehindSegment → Word3DDissolve)
    print("\n3. Testing Combined 3D Animation (6 seconds total)...")
    
    # First 3 seconds: Text3DBehindSegment
    combined_frames = []
    
    # Phase 1: Text moves behind (3 seconds)
    for i, bg_frame in enumerate(frames[:int(3.0 * fps)]):
        print(f"  Phase 1 - Frame {i+1}/{int(3.0 * fps)}...", end='\r')
        anim_frame = anim1.generate_frame(i, bg_frame)
        combined_frames.append(anim_frame)
    
    # Phase 2: Text dissolves (3 seconds)
    # Loop video if needed
    bg_frames_phase2 = []
    for i in range(int(3.0 * fps)):
        frame_idx = (int(3.0 * fps) + i) % len(frames)
        bg_frames_phase2.append(frames[frame_idx])
    
    for i, bg_frame in enumerate(bg_frames_phase2):
        print(f"  Phase 2 - Frame {i+1}/{int(3.0 * fps)}...", end='\r')
        anim_frame = anim2.generate_frame(i, bg_frame)
        combined_frames.append(anim_frame)
    
    print("\n  Combined animation complete!")
    
    # Save videos
    print("\n" + "="*60)
    print("Saving output videos...")
    print("="*60)
    
    # Save Text3DBehindSegment
    output_path_1 = "text_3d_behind_segment.mp4"
    print(f"\nSaving: {output_path_1}")
    save_video(output_frames_1, output_path_1, fps, resolution)
    
    # Save Word3DDissolve
    output_path_2 = "word_3d_dissolve.mp4"
    print(f"Saving: {output_path_2}")
    save_video(output_frames_2, output_path_2, fps, resolution)
    
    # Save Combined
    output_path_3 = "text_3d_combined_animation.mp4"
    print(f"Saving: {output_path_3}")
    save_video(combined_frames, output_path_3, fps, resolution)
    
    print("\n" + "="*60)
    print("✅ All 3D animations completed successfully!")
    print("="*60)
    print("\nOutput files:")
    print(f"  1. {output_path_1} - 3D text moving behind subject")
    print(f"  2. {output_path_2} - 3D text dissolving with particles")
    print(f"  3. {output_path_3} - Combined 6-second animation")
    print("\nFeatures demonstrated:")
    print("  • 3D text with depth layers and perspective")
    print("  • Volumetric dissolve with per-layer particle effects")
    print("  • Shadow and outline effects for depth perception")
    print("  • Smooth transition from foreground to background")


def save_video(frames, output_path, fps, resolution):
    """Save frames to video file with H.264 encoding."""
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
    
    # Convert to H.264 using FFmpeg
    import subprocess
    cmd = [
        'ffmpeg', '-y', '-i', temp_path,
        '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
        '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
        output_path
    ]
    
    subprocess.run(cmd, capture_output=True, text=True)
    
    # Remove temp file
    if os.path.exists(temp_path):
        os.remove(temp_path)
    
    print(f"  ✓ Saved with H.264 encoding")


if __name__ == "__main__":
    main()