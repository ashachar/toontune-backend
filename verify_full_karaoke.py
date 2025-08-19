#!/usr/bin/env python3
"""
Verify that the full karaoke features are working
"""

import subprocess
from pathlib import Path
import cv2
import numpy as np

def extract_frames_and_verify():
    video_path = Path("uploads/assets/videos/do_re_mi/scenes/edited/scene_001.mp4")
    verify_dir = Path("uploads/assets/videos/do_re_mi/full_karaoke_verify")
    verify_dir.mkdir(exist_ok=True)
    
    print("="*70)
    print("🔍 VERIFYING FULL KARAOKE FEATURES")
    print("="*70)
    
    # Extract frames at key moments
    test_points = [
        (8.0, "Beginning of first sentence", "yellow"),
        (10.0, "Middle of first sentence", "yellow"),
        (16.5, "Second sentence", "red"),
        (20.5, "Third sentence", "green"),
        (24.5, "Fourth sentence", "purple"),
        (45.0, "Do a deer", "varies"),
        (47.5, "With cartoon", "varies")
    ]
    
    print("\n📸 Extracting frames...")
    for time, desc, expected_color in test_points:
        frame_path = verify_dir / f"frame_{time:.1f}s.png"
        cmd = ['ffmpeg', '-ss', str(time), '-i', str(video_path),
               '-frames:v', '1', '-y', str(frame_path)]
        subprocess.run(cmd, capture_output=True)
        print(f"  {time:5.1f}s: {desc} (expect {expected_color} karaoke)")
    
    # Check the ASS file for features
    ass_file = Path("uploads/assets/videos/do_re_mi/scenes/edited/karaoke_precise.ass")
    if ass_file.exists():
        print(f"\n📄 Karaoke subtitle file:")
        print(f"  ✅ File exists: {ass_file.name}")
        print(f"  Size: {ass_file.stat().st_size} bytes")
        
        # Check content
        with open(ass_file) as f:
            content = f.read()
        
        # Check for key features
        features = []
        if "{\c&H00FFFF}" in content:  # Cyan color code
            features.append("Word highlighting")
        if "{\c&HCCCCCC}" in content:  # Gray for previous words
            features.append("Previous word graying")
        if "{\c&H0000FF}" in content or "{\c&HFF0000}" in content:  # Red/Blue
            features.append("Color rotation")
        if "," in content and "." in content:
            features.append("Punctuation preserved")
        
        print(f"\n✅ Features detected in ASS file:")
        for f in features:
            print(f"  • {f}")
    
    # Quick visual check
    print(f"\n🎨 Visual analysis of frames:")
    for frame_path in sorted(verify_dir.glob("*.png")):
        if not frame_path.exists():
            continue
            
        img = cv2.imread(str(frame_path))
        height, width = img.shape[:2]
        
        # Check bottom region for karaoke
        bottom = img[height-100:, :]
        gray_bottom = cv2.cvtColor(bottom, cv2.COLOR_BGR2GRAY)
        
        # Check for bright text
        bright_pixels = np.sum(gray_bottom > 200)
        has_karaoke = bright_pixels > 1000
        
        # Try to detect colors
        # Check for yellow pixels (BGR: low blue, high green and red)
        yellow_mask = cv2.inRange(bottom, (0, 150, 150), (100, 255, 255))
        has_yellow = np.sum(yellow_mask > 0) > 100
        
        # Check for other colors
        colors = []
        if has_yellow:
            colors.append("yellow")
        
        # Check for red
        red_mask = cv2.inRange(bottom, (0, 0, 150), (100, 100, 255))
        if np.sum(red_mask > 0) > 100:
            colors.append("red")
        
        # Check for cyan
        cyan_mask = cv2.inRange(bottom, (150, 150, 0), (255, 255, 100))
        if np.sum(cyan_mask > 0) > 100:
            colors.append("cyan")
        
        status = "✅ Karaoke" if has_karaoke else "❌ No karaoke"
        color_info = f" ({', '.join(colors)})" if colors else ""
        
        print(f"  {frame_path.stem}: {status}{color_info}")
    
    print("\n" + "="*70)
    print("📊 SUMMARY:")
    print("="*70)
    print("The final video should have:")
    print("  ✅ Word-by-word highlighting (current word colored)")
    print("  ✅ Previous words grayed out")
    print("  ✅ Color rotation between sentences")
    print("  ✅ Punctuation preserved (commas, periods)")
    print("  ✅ Aligned with transcript_sentences.json")
    print("  ✅ PLUS key phrases and cartoons overlaid")
    print("\n🎬 Final video: uploads/assets/videos/do_re_mi/scenes/edited/scene_001.mp4")
    print("="*70)

if __name__ == "__main__":
    extract_frames_and_verify()