#!/usr/bin/env python3
"""
Smart Transparency for Videos
Detects what color Kling added to fill transparent areas and removes only those pixels
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

def detect_kling_background_color(original_img_path, video_path):
    """Detect what color Kling used to fill transparent areas"""
    print("🔍 Detecting Kling background fill color...")
    
    # Load original image with transparency
    original = Image.open(original_img_path).convert('RGBA')
    orig_array = np.array(original)
    
    # Find transparent pixels in original
    transparent_mask = orig_array[:, :, 3] < 128
    
    if not np.any(transparent_mask):
        print("  No transparent pixels in original - no fill detection needed")
        return None, 0
    
    # Extract first frame from video
    cap = cv2.VideoCapture(str(video_path))
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        return None, 0
    
    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Resize frame to match original if needed
    if frame_rgb.shape[:2] != original.size[::-1]:
        frame_rgb = cv2.resize(frame_rgb, (original.width, original.height))
    
    # Get colors where original was transparent
    filled_pixels = frame_rgb[transparent_mask]
    
    if len(filled_pixels) == 0:
        return None, 0
    
    # Find the most common fill color
    # Kling typically uses pure black (0,0,0) or white (255,255,255)
    fill_color = np.median(filled_pixels, axis=0).astype(int)
    std_dev = np.std(filled_pixels, axis=0).mean()
    
    # Check if it's a uniform fill
    if std_dev < 10:
        print(f"  ✓ Detected uniform fill: RGB{tuple(fill_color)} (std: {std_dev:.1f})")
        print(f"    Kling filled {np.sum(transparent_mask)} transparent pixels")
        return tuple(fill_color), std_dev
    else:
        print(f"  ⚠️ Fill color varies: RGB{tuple(fill_color)} (std: {std_dev:.1f})")
        return tuple(fill_color), std_dev

def process_frame_smart(args):
    """Process a single frame with smart transparency"""
    frame_path, output_path, fill_color, tolerance = args
    
    img = Image.open(frame_path).convert('RGBA')
    img_array = np.array(img)
    
    if fill_color is not None:
        # Calculate color distance
        color_diff = np.abs(img_array[:, :, :3] - np.array(fill_color))
        color_distance = np.sum(color_diff, axis=2)
        
        # Make matching pixels transparent
        mask = color_distance < tolerance
        img_array[mask, 3] = 0
    
    Image.fromarray(img_array).save(output_path, 'PNG')
    return output_path

def apply_smart_transparency_to_video(video_path, output_path, original_img_path, fill_color=None):
    """Apply smart transparency to entire video"""
    
    # If no fill color provided, detect it
    if fill_color is None:
        fill_color, std_dev = detect_kling_background_color(original_img_path, video_path)
        
        if fill_color is None:
            print("  No fill color detected - keeping video as is")
            shutil.copy2(video_path, output_path)
            return output_path
    else:
        std_dev = 5  # Default for manual color
    
    # Determine tolerance based on uniformity
    if std_dev < 10:
        tolerance = 30  # Strict for uniform fills
    else:
        tolerance = 50  # Looser for varied fills
    
    print(f"🎬 Processing video with smart transparency...")
    print(f"  Fill color: RGB{fill_color}")
    print(f"  Tolerance: {tolerance}")
    
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
                fill_color,
                tolerance
            )
            for frame_file in frame_files
        ]
        
        # Process with progress
        processed_count = 0
        with ProcessPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(process_frame_smart, args) for args in process_args]
            
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
        
        print(f"  ✅ Smart transparent WebM saved: {output_path}")
        return output_path
        
    finally:
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 2:
        video_path = sys.argv[1]
        original_img = sys.argv[2]
        output_path = video_path.replace('.mp4', '_transparent.webm')
        
        apply_smart_transparency_to_video(video_path, output_path, original_img)
    else:
        print("Usage: python smart_transparency_video.py <video.mp4> <original_image.png>")