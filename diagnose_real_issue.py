#!/usr/bin/env python3
"""
Diagnose the REAL issue - why features aren't accumulating
"""

import subprocess
import shutil
import hashlib
from pathlib import Path
import time

def get_file_hash(filepath):
    """Get hash of file to track changes."""
    with open(filepath, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()[:8]

def test_step_chaining():
    """Test if steps are actually chaining correctly."""
    
    base_dir = Path("uploads/assets/videos/do_re_mi")
    video_path = base_dir / "scenes/edited/scene_001.mp4"
    original = base_dir / "scenes/original/scene_001.mp4"
    
    # Start fresh
    if original.exists():
        shutil.copy(original, video_path)
    
    print("Testing step chaining...")
    print(f"Starting hash: {get_file_hash(video_path)}")
    
    # Test 1: Add a simple overlay
    print("\n1. Adding test overlay 1...")
    temp1 = video_path.parent / "test1_temp.mp4"
    cmd = [
        'ffmpeg', '-i', str(video_path),
        '-vf', "drawtext=text='TEST1':fontsize=40:fontcolor=red:x=100:y=100:enable='between(t,0,5)'",
        '-c:v', 'libx264', '-crf', '18',
        '-y', str(temp1)
    ]
    subprocess.run(cmd, capture_output=True)
    
    if temp1.exists():
        shutil.move(temp1, video_path)
        print(f"   Hash after step 1: {get_file_hash(video_path)}")
    
    # Test 2: Add another overlay ON TOP
    print("\n2. Adding test overlay 2 on top...")
    temp2 = video_path.parent / "test2_temp.mp4"
    cmd = [
        'ffmpeg', '-i', str(video_path),  # Should have TEST1 already
        '-vf', "drawtext=text='TEST2':fontsize=40:fontcolor=blue:x=300:y=100:enable='between(t,0,5)'",
        '-c:v', 'libx264', '-crf', '18',
        '-y', str(temp2)
    ]
    subprocess.run(cmd, capture_output=True)
    
    if temp2.exists():
        shutil.move(temp2, video_path)
        print(f"   Hash after step 2: {get_file_hash(video_path)}")
    
    # Extract a frame to check
    print("\n3. Checking what's visible...")
    check_frame = video_path.parent / "chain_test.png"
    cmd = ['ffmpeg', '-ss', '2', '-i', str(video_path),
           '-frames:v', '1', '-y', str(check_frame)]
    subprocess.run(cmd, capture_output=True)
    
    # Check if both overlays are present
    import cv2
    import numpy as np
    
    img = cv2.imread(str(check_frame))
    
    # Check for red pixels (TEST1)
    red_mask = cv2.inRange(img, (0, 0, 150), (100, 100, 255))
    has_test1 = np.sum(red_mask > 0) > 100
    
    # Check for blue pixels (TEST2)
    blue_mask = cv2.inRange(img, (150, 0, 0), (255, 100, 100))
    has_test2 = np.sum(blue_mask > 0) > 100
    
    print(f"\nRESULT:")
    print(f"  TEST1 (red) present: {has_test1}")
    print(f"  TEST2 (blue) present: {has_test2}")
    
    if has_test1 and has_test2:
        print("  ✅ Chaining works! Both overlays present.")
    else:
        print("  ❌ Chaining broken! Second overlay replaced first.")
    
    return has_test1 and has_test2

def check_actual_pipeline_files():
    """Check what the actual pipeline is doing with files."""
    
    print("\n" + "="*60)
    print("CHECKING ACTUAL PIPELINE BEHAVIOR")
    print("="*60)
    
    base_dir = Path("uploads/assets/videos/do_re_mi/scenes/edited")
    
    # List all video files
    videos = list(base_dir.glob("*.mp4"))
    print(f"\nVideo files in edited directory:")
    for v in sorted(videos):
        size_mb = v.stat().st_size / (1024*1024)
        print(f"  {v.name}: {size_mb:.1f} MB")
    
    # Check for any subtitle files
    subs = list(base_dir.glob("*.ass"))
    print(f"\nSubtitle files:")
    for s in subs:
        print(f"  {s.name}")

if __name__ == "__main__":
    chaining_works = test_step_chaining()
    check_actual_pipeline_files()
    
    if chaining_works:
        print("\n✅ Basic chaining works - the issue must be in the pipeline steps themselves")
    else:
        print("\n❌ Basic chaining is broken - FFmpeg is replacing instead of layering")