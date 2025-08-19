#!/usr/bin/env python3
"""
Extract and carefully analyze frames from the final video
"""

import subprocess
from pathlib import Path
import cv2
import numpy as np
from datetime import datetime

def extract_and_analyze():
    print("="*70)
    print("🔬 DETAILED FRAME ANALYSIS")
    print("="*70)
    
    base_dir = Path("uploads/assets/videos/do_re_mi")
    final_video = base_dir / "scenes/edited/scene_001.mp4"
    
    print(f"\nVideo being analyzed: {final_video}")
    print(f"File size: {final_video.stat().st_size / (1024*1024):.1f} MB")
    print(f"Modified: {datetime.fromtimestamp(final_video.stat().st_mtime).strftime('%H:%M:%S')}")
    
    # Create analysis directory
    analysis_dir = base_dir / "karaoke_analysis"
    analysis_dir.mkdir(exist_ok=True)
    
    # Extract frames at key times
    test_times = [8, 10, 12, 15, 20, 25]
    
    print("\n📸 Extracting frames from FINAL video...")
    for time in test_times:
        frame_path = analysis_dir / f"final_{time}s.png"
        cmd = ['ffmpeg', '-ss', str(time), '-i', str(final_video),
               '-frames:v', '1', '-y', str(frame_path)]
        subprocess.run(cmd, capture_output=True)
        print(f"  Extracted: {frame_path.name}")
    
    # Also extract from test videos for comparison
    test_karaoke = base_dir / "test_karaoke_only.mp4"
    if test_karaoke.exists():
        print("\n📸 Extracting frames from TEST karaoke video...")
        for time in [8, 10, 12]:
            frame_path = analysis_dir / f"test_{time}s.png"
            cmd = ['ffmpeg', '-ss', str(time), '-i', str(test_karaoke),
                   '-frames:v', '1', '-y', str(frame_path)]
            subprocess.run(cmd, capture_output=True)
            print(f"  Extracted: {frame_path.name}")
    
    # Analyze each frame
    print("\n🔍 Frame Analysis:")
    print("-"*70)
    
    for frame_path in sorted(analysis_dir.glob("*.png")):
        if not frame_path.exists():
            continue
        
        img = cv2.imread(str(frame_path))
        height, width = img.shape[:2]
        
        # Analyze different regions
        # Bottom region (where karaoke should be)
        bottom_region = img[height-100:, :]
        
        # Check for text characteristics
        gray_bottom = cv2.cvtColor(bottom_region, cv2.COLOR_BGR2GRAY)
        
        # Look for high contrast (text vs background)
        _, binary = cv2.threshold(gray_bottom, 200, 255, cv2.THRESH_BINARY)
        white_pixels = np.sum(binary > 0)
        
        # Check for structured patterns (text lines)
        edges = cv2.Canny(gray_bottom, 50, 150)
        edge_pixels = np.sum(edges > 0)
        
        # Color analysis
        mean_color = np.mean(bottom_region, axis=(0,1))  # BGR
        
        print(f"\n{frame_path.name}:")
        print(f"  Bottom region brightness: {np.mean(gray_bottom):.1f}")
        print(f"  White pixels (text): {white_pixels}")
        print(f"  Edge pixels: {edge_pixels}")
        print(f"  Mean color (BGR): [{mean_color[0]:.0f}, {mean_color[1]:.0f}, {mean_color[2]:.0f}]")
        
        # Determine if karaoke is present
        has_karaoke = white_pixels > 500 or edge_pixels > 1000
        print(f"  Verdict: {'✅ KARAOKE PRESENT' if has_karaoke else '❌ NO KARAOKE'}")
    
    # Compare file hashes to ensure we're looking at the right video
    print("\n📁 File verification:")
    print("-"*70)
    
    import hashlib
    
    def get_file_hash(filepath):
        with open(filepath, 'rb') as f:
            return hashlib.md5(f.read(1024*1024)).hexdigest()[:8]  # First 1MB
    
    final_hash = get_file_hash(final_video)
    print(f"Final video hash: {final_hash}")
    
    # Check if there are multiple scene_001.mp4 files
    all_scene_001 = list(base_dir.rglob("**/scene_001.mp4"))
    print(f"\nAll scene_001.mp4 files found:")
    for f in all_scene_001:
        rel_path = f.relative_to(base_dir)
        size_mb = f.stat().st_size / (1024*1024)
        print(f"  {rel_path}: {size_mb:.1f} MB")
    
    print("\n" + "="*70)
    print("💡 CONCLUSION:")
    print("-"*70)
    print(f"Frames saved to: {analysis_dir}")
    print("Please visually check these frames to see if karaoke is visible.")
    print("="*70)

if __name__ == "__main__":
    extract_and_analyze()