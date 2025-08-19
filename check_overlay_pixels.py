#!/usr/bin/env python3
"""
Check if overlays are actually visible in the frames
"""

import cv2
import numpy as np
from pathlib import Path

def check_for_text_overlay(image_path, expected_text_region):
    """Check if there's text in the expected region by looking for non-video pixels."""
    img = cv2.imread(str(image_path))
    if img is None:
        return False, "Could not load image"
    
    height, width = img.shape[:2]
    
    # For phrase 1 at (680, 450)
    if "phrase_1" in image_path.name:
        # Check region around x=680, y=450
        roi = img[440:470, 660:760]  # Region of interest
        # Look for white/bright pixels (text)
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        bright_pixels = np.sum(gray_roi > 200)
        total_pixels = roi.shape[0] * roi.shape[1]
        bright_ratio = bright_pixels / total_pixels
        
        return bright_ratio > 0.05, f"Bright pixel ratio: {bright_ratio:.3f}"
    
    # For phrase 2 at (50, 50) - playful yellow
    elif "phrase_2" in image_path.name:
        roi = img[40:70, 30:200]
        # Look for yellow pixels
        # Yellow in BGR is high Blue and Green, low Red
        yellow_mask = cv2.inRange(roi, (0, 150, 150), (100, 255, 255))
        yellow_pixels = np.sum(yellow_mask > 0)
        total_pixels = roi.shape[0] * roi.shape[1]
        yellow_ratio = yellow_pixels / total_pixels
        
        return yellow_ratio > 0.01, f"Yellow pixel ratio: {yellow_ratio:.3f}"
    
    # For cartoons - look for the spring.png overlay
    elif "cartoon" in image_path.name:
        # Cartoons should have distinct colored regions
        # Look for non-video content by checking color variance
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Check if there are distinct color regions (cartoon overlays)
        unique_hues = len(np.unique(hsv[:,:,0]))
        
        return unique_hues > 100, f"Unique hues: {unique_hues}"
    
    return False, "Unknown frame type"

def main():
    debug_dir = Path("uploads/assets/videos/do_re_mi/debug_frames")
    
    print("🔍 Checking for overlays in extracted frames:")
    print("-" * 50)
    
    for frame_path in sorted(debug_dir.glob("*.png")):
        has_overlay, details = check_for_text_overlay(frame_path, None)
        
        status = "✓ OVERLAY DETECTED" if has_overlay else "✗ NO OVERLAY"
        print(f"{frame_path.name:30} {status:20} {details}")
    
    # Also check if test overlay has the TEST text
    print("\nChecking test overlay video frame:")
    test_frame = debug_dir / "test_frame.png"
    if (debug_dir / "test_overlay.mp4").exists():
        # Extract a frame from test video
        import subprocess
        subprocess.run([
            "ffmpeg", "-i", str(debug_dir / "test_overlay.mp4"),
            "-ss", "2", "-frames:v", "1",
            "-y", str(test_frame)
        ], capture_output=True)
        
        if test_frame.exists():
            img = cv2.imread(str(test_frame))
            # Look for red TEST text at (100, 100)
            roi = img[90:130, 90:200]
            red_mask = cv2.inRange(roi, (0, 0, 150), (100, 100, 255))
            red_pixels = np.sum(red_mask > 0)
            if red_pixels > 100:
                print("✓ TEST text found in test overlay - FFmpeg is working!")
            else:
                print("✗ TEST text not found - FFmpeg might have issues")

if __name__ == "__main__":
    main()