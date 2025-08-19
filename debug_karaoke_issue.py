#!/usr/bin/env python3
"""
Debug why karaoke isn't appearing in final video
"""

import subprocess
from pathlib import Path
import cv2
import numpy as np

def test_karaoke_alone():
    """Test if karaoke works by itself."""
    print("="*70)
    print("🔍 DEBUGGING KARAOKE ISSUE")
    print("="*70)
    
    base_dir = Path("uploads/assets/videos/do_re_mi")
    original = base_dir / "scenes/original/scene_001.mp4"
    ass_file = base_dir / "scenes/edited/karaoke_precise.ass"
    
    # 1. Check if ASS file exists
    if ass_file.exists():
        print(f"✅ ASS file exists: {ass_file.stat().st_size} bytes")
    else:
        print(f"❌ ASS file NOT found at {ass_file}")
        return
    
    # 2. Test karaoke ALONE (no other overlays)
    print("\n1️⃣ Testing karaoke ALONE...")
    test_output = base_dir / "test_karaoke_only.mp4"
    
    cmd = [
        'ffmpeg', '-i', str(original),
        '-vf', f"ass={ass_file}",
        '-c:v', 'libx264', '-crf', '18',
        '-t', '30',  # Just first 30 seconds
        '-y', str(test_output)
    ]
    
    print(f"  Command: {' '.join(cmd[:5])}...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"  ✅ Karaoke-only video created: {test_output.name}")
        
        # Extract frame to check
        frame_path = base_dir / "test_karaoke_frame.png"
        subprocess.run(['ffmpeg', '-ss', '10', '-i', str(test_output),
                       '-frames:v', '1', '-y', str(frame_path)], capture_output=True)
        
        # Check if karaoke is visible
        img = cv2.imread(str(frame_path))
        height = img.shape[0]
        bottom = img[height-100:, :]
        gray = cv2.cvtColor(bottom, cv2.COLOR_BGR2GRAY)
        if np.mean(gray) > 30:
            print(f"  ✅ Karaoke IS visible in test video!")
        else:
            print(f"  ❌ No karaoke visible in test video")
    else:
        print(f"  ❌ Failed: {result.stderr[:200]}")
    
    # 3. Test with subtitles filter instead
    print("\n2️⃣ Testing with subtitles filter...")
    test_output2 = base_dir / "test_subtitles_filter.mp4"
    
    cmd = [
        'ffmpeg', '-i', str(original),
        '-vf', f"subtitles={ass_file}",
        '-c:v', 'libx264', '-crf', '18',
        '-t', '30',
        '-y', str(test_output2)
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"  ✅ Subtitles filter worked: {test_output2.name}")
    else:
        print(f"  ❌ Subtitles filter failed: {result.stderr[:200]}")
    
    # 4. Check what the single-pass pipeline is actually doing
    print("\n3️⃣ Checking single-pass pipeline command...")
    
    # Simulate what the pipeline does
    cartoon_asset = Path("cartoon-test/spring.png")
    test_combined = base_dir / "test_combined.mp4"
    
    if cartoon_asset.exists():
        # Build the same filter_complex as the pipeline
        filter_complex = f"[0:v]ass={ass_file}[with_karaoke]"
        
        cmd = [
            'ffmpeg', '-i', str(original),
            '-filter_complex', filter_complex,
            '-map', '[with_karaoke]',
            '-map', '0:a?',
            '-c:v', 'libx264', '-crf', '18',
            '-t', '15',
            '-y', str(test_combined)
        ]
        
        print(f"  Testing filter_complex with ass...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"  ✅ Filter_complex with ass worked")
        else:
            print(f"  ❌ Filter_complex failed: {result.stderr[:200]}")
    
    # 5. Check the actual final video
    print("\n4️⃣ Checking the ACTUAL final video...")
    final_video = base_dir / "scenes/edited/scene_001.mp4"
    
    if final_video.exists():
        # Extract frames at karaoke times
        for time in [8, 10, 15, 20, 25]:
            frame_path = base_dir / f"final_check_{time}s.png"
            subprocess.run(['ffmpeg', '-ss', str(time), '-i', str(final_video),
                           '-frames:v', '1', '-y', str(frame_path)], capture_output=True)
            
            # Check for karaoke
            if frame_path.exists():
                img = cv2.imread(str(frame_path))
                height = img.shape[0]
                bottom = img[height-100:, :]
                gray = cv2.cvtColor(bottom, cv2.COLOR_BGR2GRAY)
                bright = np.mean(gray)
                
                has_karaoke = bright > 30
                status = "✅ HAS KARAOKE" if has_karaoke else "❌ NO KARAOKE"
                print(f"  {time:3}s: {status} (brightness: {bright:.1f})")
    
    print("\n" + "="*70)
    print("📊 DIAGNOSIS:")
    print("-"*70)
    
    # Check if the ASS file timing is correct
    print("\n5️⃣ Checking ASS file timing...")
    with open(ass_file) as f:
        lines = f.readlines()
    
    # Find first few dialogue lines
    dialogues = [l for l in lines if l.startswith("Dialogue:")][:5]
    print("  First dialogues in ASS:")
    for d in dialogues:
        # Extract timing
        parts = d.split(',')
        if len(parts) > 2:
            start = parts[1]
            end = parts[2]
            print(f"    {start} -> {end}")
    
    print("\n" + "="*70)

if __name__ == "__main__":
    test_karaoke_alone()