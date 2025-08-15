#!/usr/bin/env python3
"""
Green Screen Removal for Videos
Removes green screen background from videos using chroma keying
"""

import cv2
import numpy as np
from PIL import Image
import subprocess
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import tempfile
import shutil

def remove_green_screen_from_frame(args):
    """Remove green screen from a single frame"""
    frame_path, output_path, threshold = args
    
    # Load image
    img = cv2.imread(str(frame_path))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Convert to HSV for better green detection
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Define green color range in HSV
    # Hue: 40-80 (green range), Saturation: 40-255, Value: 40-255
    lower_green = np.array([40, 40, 40])
    upper_green = np.array([80, 255, 255])
    
    # Create mask for green pixels
    mask = cv2.inRange(hsv, lower_green, upper_green)
    
    # Invert mask (green becomes transparent)
    mask_inv = cv2.bitwise_not(mask)
    
    # Apply some morphology to clean up edges
    kernel = np.ones((3,3), np.uint8)
    mask_inv = cv2.morphologyEx(mask_inv, cv2.MORPH_OPEN, kernel)
    mask_inv = cv2.morphologyEx(mask_inv, cv2.MORPH_CLOSE, kernel)
    
    # Apply Gaussian blur to soften edges
    mask_inv = cv2.GaussianBlur(mask_inv, (5,5), 0)
    
    # Create RGBA image
    img_rgba = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2RGBA)
    img_rgba[:, :, 3] = mask_inv
    
    # Save as PNG with transparency
    Image.fromarray(img_rgba).save(output_path, 'PNG')
    return output_path

def apply_green_screen_removal_to_video(video_path, output_path):
    """Apply green screen removal to entire video"""
    
    print("🎬 Removing green screen from video...")
    
    # Create temp directory
    temp_dir = Path(tempfile.mkdtemp())
    frames_dir = temp_dir / "frames"
    frames_dir.mkdir()
    
    try:
        # Extract frames
        print("  📽️ Extracting frames...")
        cmd = [
            'ffmpeg', '-i', str(video_path),
            '-vf', 'fps=30',
            str(frames_dir / 'frame_%04d.png'),
            '-loglevel', 'error'
        ]
        subprocess.run(cmd, check=True)
        
        # Count frames
        frame_files = sorted(frames_dir.glob('frame_*.png'))
        total_frames = len(frame_files)
        print(f"  📊 Processing {total_frames} frames...")
        
        # Process frames in parallel
        processed_dir = temp_dir / "processed"
        processed_dir.mkdir()
        
        # Prepare arguments
        process_args = [
            (
                str(frame_file),
                str(processed_dir / frame_file.name),
                30  # threshold
            )
            for frame_file in frame_files
        ]
        
        # Process with progress
        processed_count = 0
        with ProcessPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(remove_green_screen_from_frame, args) for args in process_args]
            
            for future in as_completed(futures):
                processed_count += 1
                if processed_count % 100 == 0 or processed_count == total_frames:
                    print(f"    Processed {processed_count}/{total_frames} frames...")
        
        # Create WebM with transparency
        print("  🎥 Creating WebM with transparency...")
        cmd = [
            'ffmpeg',
            '-framerate', '30',
            '-i', str(processed_dir / 'frame_%04d.png'),
            '-c:v', 'libvpx-vp9',
            '-pix_fmt', 'yuva420p',
            '-b:v', '2M',
            '-auto-alt-ref', '0',
            str(output_path),
            '-y',
            '-loglevel', 'error'
        ]
        subprocess.run(cmd, check=True)
        
        print(f"  ✅ Green screen removed: {output_path}")
        return output_path
        
    finally:
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)

def create_green_screen_grid(image_paths, output_path, grid_size=(3, 3)):
    """Create a grid with green screen background"""
    from PIL import Image
    
    cols, rows = grid_size
    images = []
    
    print("Creating green screen grid...")
    
    # Load images
    for img_path in image_paths:
        img = Image.open(img_path).convert('RGBA')
        images.append(img)
    
    # Find max dimensions
    max_width = max(img.width for img in images)
    max_height = max(img.height for img in images)
    
    # Ensure even dimensions
    cell_width = max_width + (max_width % 2)
    cell_height = max_height + (max_height % 2)
    
    # Create grid with green screen
    grid_width = cell_width * cols
    grid_height = cell_height * rows
    
    # Chroma key green
    green_screen = (0, 255, 0, 255)
    grid = Image.new('RGBA', (grid_width, grid_height), green_screen)
    
    # Place images
    for i, img in enumerate(images):
        if i >= cols * rows:
            break
        row = i // cols
        col = i % cols
        x = col * cell_width + (cell_width - img.width) // 2
        y = row * cell_height + (cell_height - img.height) // 2
        grid.paste(img, (x, y), img)
    
    grid.save(output_path)
    print(f"✅ Green screen grid saved: {output_path}")
    return output_path

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        if sys.argv[1].endswith('.mp4'):
            # Remove green screen from video
            video_path = sys.argv[1]
            output_path = video_path.replace('.mp4', '_no_green.webm')
            apply_green_screen_removal_to_video(video_path, output_path)
        else:
            print("Usage: python green_screen_removal.py <video.mp4>")