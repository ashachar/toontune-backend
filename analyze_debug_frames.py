#!/usr/bin/env python3
"""
Analyze the debug frames to see what features are visible
"""

import cv2
import numpy as np
from pathlib import Path

def analyze_frame(frame_path, description):
    """Analyze a frame for visible features."""
    if not frame_path.exists():
        return f"  ❌ {description}: File not found"
    
    img = cv2.imread(str(frame_path))
    if img is None:
        return f"  ❌ {description}: Could not load"
    
    height, width = img.shape[:2]
    
    # Check different regions for features
    results = []
    
    # Check for karaoke at bottom (bright text in bottom 100 pixels)
    bottom_region = img[height-100:, :]
    gray_bottom = cv2.cvtColor(bottom_region, cv2.COLOR_BGR2GRAY)
    bright_pixels_bottom = np.sum(gray_bottom > 200)
    if bright_pixels_bottom > 1000:  # Significant bright pixels
        results.append("KARAOKE")
    
    # Check for text in top regions (phrases)
    # Top-left for phrase 2
    top_left = img[30:80, 30:200]
    # Check for yellow pixels (phrase 2)
    yellow_mask = cv2.inRange(top_left, (0, 150, 150), (100, 255, 255))
    if np.sum(yellow_mask > 0) > 100:
        results.append("PHRASE2(yellow@top-left)")
    
    # Top-right for phrase 1  
    top_right = img[430:480, 650:850]
    gray_top_right = cv2.cvtColor(top_right, cv2.COLOR_BGR2GRAY)
    if np.sum(gray_top_right > 200) > 500:
        results.append("PHRASE1(white@top-right)")
    
    # Check for cartoon overlay (unique colors in middle region)
    middle = img[height//3:2*height//3, width//3:2*width//3]
    unique_colors = len(np.unique(middle.reshape(-1, 3), axis=0))
    if unique_colors > 5000:  # Many unique colors suggest overlay
        results.append("CARTOON")
    
    status = " + ".join(results) if results else "NOTHING"
    return f"  {description:30} → {status}"

def main():
    debug_dir = Path("uploads/assets/videos/do_re_mi/ultrathink_debug")
    
    print("="*70)
    print("🔍 FRAME ANALYSIS - What's Actually Visible")
    print("="*70)
    
    frames = [
        ("after_karaoke.png", "After Karaoke (25s)"),
        ("after_phrases_1.png", "After Phrases (11.5s)"),
        ("after_phrases_2.png", "After Phrases (23s)"),
        ("after_cartoons_1.png", "After Cartoons (47.5s)"),
        ("after_cartoons_2.png", "After Cartoons (51.5s)"),
        ("test_multi_p1.png", "Test Multi P1 (11.5s)"),
        ("test_multi_p2.png", "Test Multi P2 (23s)")
    ]
    
    print("\nFRAME CONTENTS:")
    print("-"*70)
    for filename, desc in frames:
        result = analyze_frame(debug_dir / filename, desc)
        print(result)
    
    # Also check the actual final video
    print("\nFINAL VIDEO FRAMES:")
    print("-"*70)
    
    final_check_dir = Path("uploads/assets/videos/do_re_mi/final_check")
    if final_check_dir.exists():
        for frame_path in sorted(final_check_dir.glob("frame_*.png")):
            time = frame_path.stem.replace("frame_", "").replace("s", "")
            result = analyze_frame(frame_path, f"Final at {time}")
            print(result)
    
    print("\n" + "="*70)
    print("LEGEND:")
    print("  KARAOKE = Bright text at bottom")
    print("  PHRASE1 = White text top-right")
    print("  PHRASE2 = Yellow text top-left")
    print("  CARTOON = Complex overlay in middle")
    print("="*70)

if __name__ == "__main__":
    main()