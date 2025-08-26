#!/usr/bin/env python3
"""
Video Frame Grid Generator

Creates a grid visualization of video frames sampled at 1-second intervals.
Each row represents 30 seconds (30 frames), making it easy to visually identify scene changes.
"""

import cv2
import numpy as np
from pathlib import Path
import argparse
import sys


def create_frame_grid(video_path, output_path=None, sample_interval=1.0, row_duration=10.0, frame_scale=0.15):
    """
    Create a grid of video frames for visual scene detection
    
    Args:
        video_path: Path to input video
        output_path: Path for output image (auto-generated if None)
        sample_interval: Seconds between sampled frames (default: 1.0)
        row_duration: Seconds per row (default: 30.0)
        frame_scale: Scale factor for individual frames (default: 0.1)
    """
    video_path = Path(video_path)
    if not video_path.exists():
        print(f"❌ Video not found: {video_path}")
        return None
    
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps if fps > 0 else 0
    
    print(f"📹 Video: {width}x{height}, {fps:.1f} fps, {duration:.1f}s")
    
    # Calculate grid dimensions
    frames_per_row = int(row_duration / sample_interval)  # 30 frames per row for 1s interval
    frame_interval = int(fps * sample_interval)  # Frames to skip
    
    # Calculate scaled frame dimensions
    scaled_width = int(width * frame_scale)
    scaled_height = int(height * frame_scale)
    
    print(f"📐 Grid: {frames_per_row} frames per row, {scaled_width}x{scaled_height} per frame")
    
    # Collect frames
    frames = []
    frame_count = 0
    sampled_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Sample at intervals
        if frame_count % frame_interval == 0:
            # Resize frame
            frame_small = cv2.resize(frame, (scaled_width, scaled_height))
            frames.append(frame_small)
            sampled_count += 1
            
            if sampled_count % 10 == 0:
                print(f"  Sampled {sampled_count} frames...")
        
        frame_count += 1
    
    cap.release()
    
    if not frames:
        print("❌ No frames sampled")
        return None
    
    print(f"✅ Sampled {len(frames)} frames")
    
    # Calculate grid layout
    num_rows = (len(frames) + frames_per_row - 1) // frames_per_row
    grid_width = frames_per_row * scaled_width
    grid_height = num_rows * scaled_height
    
    print(f"🖼️ Creating grid: {grid_width}x{grid_height} ({num_rows} rows)")
    
    # Create grid image
    grid = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)
    
    # Place frames in grid
    for i, frame in enumerate(frames):
        row = i // frames_per_row
        col = i % frames_per_row
        
        y1 = row * scaled_height
        y2 = y1 + scaled_height
        x1 = col * scaled_width
        x2 = x1 + scaled_width
        
        grid[y1:y2, x1:x2] = frame
        
        # Add thin border between frames
        if col > 0:  # Vertical lines
            grid[y1:y2, x1:x1+1] = (32, 32, 32)
        if row > 0:  # Horizontal lines
            grid[y1:y1+1, x1:x2] = (32, 32, 32)
    
    # Generate output path
    if output_path is None:
        output_path = video_path.parent / f"{video_path.stem}_frame_grid.jpg"
    else:
        output_path = Path(output_path)
    
    # Save grid
    cv2.imwrite(str(output_path), grid)
    print(f"💾 Grid saved to: {output_path}")
    
    # Also save high-res version if grid is large
    if grid_width > 3000 or grid_height > 2000:
        # Compress for viewing
        quality = 95
        cv2.imwrite(str(output_path), grid, [cv2.IMWRITE_JPEG_QUALITY, quality])
        print(f"   Compressed to JPEG quality {quality}")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description='Generate a grid visualization of video frames',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (1 frame per second, 30 seconds per row)
  python video_frame_grid.py video.mp4
  
  # Sample every 2 seconds
  python video_frame_grid.py video.mp4 --interval 2
  
  # 60 seconds per row (for longer videos)
  python video_frame_grid.py video.mp4 --row-duration 60
  
  # Larger frame thumbnails
  python video_frame_grid.py video.mp4 --scale 0.2
        """
    )
    
    parser.add_argument('input', help='Input video file')
    parser.add_argument('-o', '--output', help='Output image file (default: <video>_frame_grid.jpg)')
    parser.add_argument('-i', '--interval', type=float, default=1.0,
                       help='Seconds between sampled frames (default: 1.0)')
    parser.add_argument('-r', '--row-duration', type=float, default=10.0,
                       help='Seconds per row (default: 10.0)')
    parser.add_argument('-s', '--scale', type=float, default=0.1,
                       help='Scale factor for frames (default: 0.1)')
    
    args = parser.parse_args()
    
    # Create grid
    output_path = create_frame_grid(
        args.input,
        args.output,
        args.interval,
        args.row_duration,
        args.scale
    )
    
    if output_path and output_path.exists():
        print(f"\n✅ Success! Grid created at: {output_path}")
        
        # Try to open the image
        import platform
        system = platform.system()
        try:
            if system == 'Darwin':  # macOS
                import subprocess
                subprocess.run(['open', str(output_path)])
                print("📂 Opening image...")
            elif system == 'Windows':
                import os
                os.startfile(str(output_path))
            else:  # Linux
                import subprocess
                subprocess.run(['xdg-open', str(output_path)])
        except Exception as e:
            print(f"ℹ️ Could not auto-open image: {e}")
            print(f"   Please open manually: {output_path}")
        
        return 0
    else:
        return 1


if __name__ == '__main__':
    sys.exit(main())