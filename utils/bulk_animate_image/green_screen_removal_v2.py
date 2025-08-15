#!/usr/bin/env python3
"""
Green Screen Removal V2 - Optimized for edge quality
Removes green screen with better edge handling and color spill suppression
"""

import cv2
import numpy as np
from PIL import Image
import subprocess
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import tempfile
import shutil

def remove_green_screen_from_frame_v2(args):
    """Remove green screen with optimized edge handling"""
    frame_path, output_path = args
    
    # Load image
    img = cv2.imread(str(frame_path))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Convert to HSV for green detection
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Create multiple masks for different green shades
    masks = []
    
    # Core green pixels (high confidence)
    lower = np.array([50, 100, 100])
    upper = np.array([70, 255, 255])
    masks.append(cv2.inRange(hsv, lower, upper))
    
    # Broader green range
    lower = np.array([40, 40, 40])
    upper = np.array([80, 255, 255])
    masks.append(cv2.inRange(hsv, lower, upper))
    
    # Very light greens (catches edges)
    lower = np.array([35, 25, 25])
    upper = np.array([85, 255, 255])
    mask_light = cv2.inRange(hsv, lower, upper)
    
    # Also check for greenish pixels in RGB space
    # Where green channel is significantly higher than red and blue
    b, g, r = cv2.split(img)
    green_dominant = ((g > r * 1.3) & (g > b * 1.3) & (g > 50)).astype(np.uint8) * 255
    
    # Combine all masks
    mask_combined = masks[0]
    for mask in masks[1:]:
        mask_combined = cv2.bitwise_or(mask_combined, mask)
    mask_combined = cv2.bitwise_or(mask_combined, green_dominant)
    
    # Clean up mask with morphology
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    
    # Remove noise
    mask_cleaned = cv2.morphologyEx(mask_combined, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Fill small holes
    mask_cleaned = cv2.morphologyEx(mask_cleaned, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    # Erode edges slightly to remove green fringe
    kernel_erode = np.ones((2, 2), np.uint8)
    mask_eroded = cv2.erode(mask_cleaned, kernel_erode, iterations=1)
    
    # Create alpha channel with smooth edges
    # Use distance transform for smooth falloff
    dist = cv2.distanceTransform(255 - mask_eroded, cv2.DIST_L2, 3)
    
    # Normalize distance for alpha blending
    dist = np.clip(dist * 50, 0, 255).astype(np.uint8)
    
    # Invert: 255 = opaque, 0 = transparent
    alpha = 255 - mask_eroded
    
    # Smooth the alpha channel
    alpha = cv2.GaussianBlur(alpha, (3, 3), 0)
    
    # Color correction: reduce green spill
    img_corrected = img_rgb.copy()
    
    # Find edge pixels (partially transparent)
    edge_mask = (alpha > 20) & (alpha < 235)
    
    if np.any(edge_mask):
        # Calculate green excess for edge pixels
        r_channel = img_corrected[:,:,0].astype(np.float32)
        g_channel = img_corrected[:,:,1].astype(np.float32)
        b_channel = img_corrected[:,:,2].astype(np.float32)
        
        # Green spill amount
        green_spill = np.maximum(0, g_channel - np.maximum(r_channel, b_channel))
        
        # Suppress green in edge areas
        suppression = 0.6
        g_channel[edge_mask] -= green_spill[edge_mask] * suppression
        
        # Rebuild image
        img_corrected[:,:,1] = np.clip(g_channel, 0, 255).astype(np.uint8)
    
    # Create RGBA image
    img_rgba = cv2.cvtColor(img_corrected, cv2.COLOR_RGB2RGBA)
    img_rgba[:, :, 3] = alpha
    
    # Save with transparency
    Image.fromarray(img_rgba).save(output_path, 'PNG')
    return output_path

def apply_green_screen_removal_v2(video_path, output_path):
    """Apply green screen removal V2 to video"""
    
    print("🎬 Removing green screen (V2 algorithm)...")
    
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
            (str(frame_file), str(processed_dir / frame_file.name))
            for frame_file in frame_files
        ]
        
        # Process with progress
        processed_count = 0
        with ProcessPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(remove_green_screen_from_frame_v2, args) 
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
        
        print(f"  ✅ Green screen removed: {output_path}")
        return output_path
        
    finally:
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
        output_path = video_path.replace('.mp4', '_no_green_v2.webm')
        apply_green_screen_removal_v2(video_path, output_path)