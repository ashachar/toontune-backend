#!/usr/bin/env python3
"""
Improved Green Screen Removal for Videos
Enhanced chroma keying with edge refinement and color spill suppression
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

def remove_green_screen_from_frame_improved(args):
    """Remove green screen from a single frame with improved edge handling"""
    frame_path, output_path = args
    
    # Load image
    img = cv2.imread(str(frame_path))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Convert to HSV for better green detection
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Multi-pass green detection with different thresholds
    # Pass 1: Core green pixels (strict)
    lower_green1 = np.array([45, 50, 50])
    upper_green1 = np.array([75, 255, 255])
    mask1 = cv2.inRange(hsv, lower_green1, upper_green1)
    
    # Pass 2: Catch edge greens (wider range)
    lower_green2 = np.array([35, 30, 30])
    upper_green2 = np.array([85, 255, 255])
    mask2 = cv2.inRange(hsv, lower_green2, upper_green2)
    
    # Pass 3: Pure green detection (for very bright greens)
    # Check in RGB space for pure green
    green_dominance = img[:,:,1].astype(np.float32) - np.maximum(img[:,:,0], img[:,:,2])
    mask3 = (green_dominance > 30).astype(np.uint8) * 255
    
    # Combine masks
    mask_combined = cv2.bitwise_or(mask1, mask2)
    mask_combined = cv2.bitwise_or(mask_combined, mask3)
    
    # Clean up the mask with morphology
    # First remove noise
    kernel_small = np.ones((2,2), np.uint8)
    mask_cleaned = cv2.morphologyEx(mask_combined, cv2.MORPH_OPEN, kernel_small)
    
    # Fill small holes
    kernel_medium = np.ones((3,3), np.uint8)
    mask_cleaned = cv2.morphologyEx(mask_cleaned, cv2.MORPH_CLOSE, kernel_medium)
    
    # Erode mask slightly to remove green fringe (shrink inward by 1-2 pixels)
    kernel_erode = np.ones((2,2), np.uint8)
    mask_eroded = cv2.erode(mask_cleaned, kernel_erode, iterations=1)
    
    # Create feathered edges for smooth transitions
    # Distance transform for edge detection
    dist_transform = cv2.distanceTransform(255 - mask_eroded, cv2.DIST_L2, 5)
    
    # Create feathering zone (2-4 pixels from edge)
    feather_zone = np.zeros_like(mask_eroded, dtype=np.float32)
    feather_width = 3
    feather_mask = (dist_transform > 0) & (dist_transform <= feather_width)
    feather_zone[feather_mask] = dist_transform[feather_mask] / feather_width
    
    # Combine solid and feathered areas
    alpha_channel = np.ones_like(mask_eroded, dtype=np.float32)
    alpha_channel[mask_eroded == 255] = 0  # Fully transparent where green
    alpha_channel[feather_mask] = feather_zone[feather_mask]  # Gradual transparency at edges
    
    # Invert for proper alpha (255 = opaque, 0 = transparent)
    alpha_channel = (1 - alpha_channel) * 255
    alpha_channel = alpha_channel.astype(np.uint8)
    
    # Apply bilateral filter for edge refinement (preserves edges while smoothing)
    # Use bilateral filter instead of guided filter to avoid dependency
    alpha_refined = cv2.bilateralFilter(
        alpha_channel,
        d=5,  # Diameter of pixel neighborhood
        sigmaColor=50,  # Filter sigma in color space
        sigmaSpace=50   # Filter sigma in coordinate space
    )
    
    # Color spill suppression - reduce green channel near edges
    img_despilled = img_rgb.copy().astype(np.float32)
    
    # Find pixels near edges (where feathering occurs)
    edge_region = (alpha_refined > 10) & (alpha_refined < 245)
    
    # Suppress green channel in edge regions
    if np.any(edge_region):
        # Calculate green excess
        green_excess = np.maximum(0, img_despilled[:,:,1] - np.maximum(img_despilled[:,:,0], img_despilled[:,:,2]))
        
        # Reduce green channel by the excess amount in edge regions
        suppression_factor = 0.7  # How much to suppress (0 = no suppression, 1 = full)
        img_despilled[:,:,1][edge_region] -= green_excess[edge_region] * suppression_factor
        
        # Ensure values stay in valid range
        img_despilled = np.clip(img_despilled, 0, 255)
    
    img_despilled = img_despilled.astype(np.uint8)
    
    # Create RGBA image
    img_rgba = cv2.cvtColor(img_despilled, cv2.COLOR_RGB2RGBA)
    img_rgba[:, :, 3] = alpha_refined
    
    # Additional edge cleanup: check for isolated green pixels
    # This catches single green pixels that might be missed
    for y in range(1, img_rgba.shape[0] - 1):
        for x in range(1, img_rgba.shape[1] - 1):
            if img_rgba[y, x, 3] > 200:  # If pixel is mostly opaque
                # Check if surrounded by transparent pixels
                surrounding_alpha = img_rgba[y-1:y+2, x-1:x+2, 3]
                if np.mean(surrounding_alpha) < 100:  # Mostly transparent surroundings
                    # Check if this pixel is greenish
                    pixel = img_rgb[y, x]
                    if pixel[1] > pixel[0] * 1.2 and pixel[1] > pixel[2] * 1.2:
                        img_rgba[y, x, 3] = 0  # Make it transparent
    
    # Save as PNG with transparency
    Image.fromarray(img_rgba).save(output_path, 'PNG')
    return output_path

def apply_green_screen_removal_improved(video_path, output_path):
    """Apply improved green screen removal to entire video"""
    
    print("🎬 Removing green screen from video (improved algorithm)...")
    
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
        print(f"  📊 Processing {total_frames} frames with improved edge handling...")
        
        # Process frames in parallel
        processed_dir = temp_dir / "processed"
        processed_dir.mkdir()
        
        # Prepare arguments
        process_args = [
            (
                str(frame_file),
                str(processed_dir / frame_file.name)
            )
            for frame_file in frame_files
        ]
        
        # Process with progress
        processed_count = 0
        with ProcessPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(remove_green_screen_from_frame_improved, args) 
                      for args in process_args]
            
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
        
        print(f"  ✅ Green screen removed (improved): {output_path}")
        return output_path
        
    finally:
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        if sys.argv[1].endswith('.mp4'):
            # Remove green screen from video
            video_path = sys.argv[1]
            output_path = video_path.replace('.mp4', '_no_green_improved.webm')
            apply_green_screen_removal_improved(video_path, output_path)
        else:
            print("Usage: python green_screen_removal_improved.py <video.mp4>")