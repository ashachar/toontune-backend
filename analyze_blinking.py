"""
Analyze frames to detect and understand blinking issue
"""

import cv2
import numpy as np
from pathlib import Path

def analyze_frames():
    frame_dir = Path("outputs/blinking_analysis")
    frames = sorted(frame_dir.glob("frame_*.jpg"))
    
    print(f"Analyzing {len(frames)} frames...")
    print("=" * 60)
    
    # Load all frames
    loaded_frames = []
    for frame_path in frames:
        frame = cv2.imread(str(frame_path))
        loaded_frames.append(frame)
        
    # Analyze text region (center area where text appears)
    text_region_y1, text_region_y2 = 300, 420  # Approximate text area
    text_region_x1, text_region_x2 = 400, 880
    
    # Compare consecutive frames
    for i in range(len(loaded_frames) - 1):
        frame1 = loaded_frames[i]
        frame2 = loaded_frames[i + 1]
        
        # Extract text regions
        region1 = frame1[text_region_y1:text_region_y2, text_region_x1:text_region_x2]
        region2 = frame2[text_region_y1:text_region_y2, text_region_x1:text_region_x2]
        
        # Calculate difference
        diff = cv2.absdiff(region1, region2)
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        
        # Check for significant changes (blinking)
        mean_diff = np.mean(diff_gray)
        max_diff = np.max(diff_gray)
        
        # Count bright pixels (text)
        _, thresh1 = cv2.threshold(cv2.cvtColor(region1, cv2.COLOR_BGR2GRAY), 100, 255, cv2.THRESH_BINARY)
        _, thresh2 = cv2.threshold(cv2.cvtColor(region2, cv2.COLOR_BGR2GRAY), 100, 255, cv2.THRESH_BINARY)
        
        bright_pixels1 = np.sum(thresh1 > 0)
        bright_pixels2 = np.sum(thresh2 > 0)
        brightness_change = abs(bright_pixels2 - bright_pixels1)
        
        print(f"Frame {i:02d} -> {i+1:02d}:")
        print(f"  Mean diff: {mean_diff:.2f}, Max diff: {max_diff}")
        print(f"  Bright pixels: {bright_pixels1} -> {bright_pixels2} (change: {brightness_change})")
        
        if brightness_change > 5000:  # Significant change in text visibility
            print(f"  ⚠️ BLINKING DETECTED! Text visibility changed significantly")
            
        # Save comparison image for visual inspection
        if i in [0, 5, 10, 15, 20, 25]:
            comparison = np.hstack([region1, region2, cv2.cvtColor(diff_gray, cv2.COLOR_GRAY2BGR)])
            cv2.imwrite(f"outputs/blinking_analysis/comparison_{i:02d}.jpg", comparison)
            print(f"  Saved comparison image: comparison_{i:02d}.jpg")
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("Blinking appears to be caused by text appearing/disappearing between frames")
    print("This suggests the animation is not maintaining consistent text visibility")

if __name__ == "__main__":
    analyze_frames()