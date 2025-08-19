#!/usr/bin/env python3
"""
Final verification - extract frames and check all features
"""

import subprocess
from pathlib import Path
from datetime import datetime

def extract_and_verify():
    video_path = Path("uploads/assets/videos/do_re_mi/scenes/edited/scene_001.mp4")
    output_dir = Path("uploads/assets/videos/do_re_mi/final_check")
    output_dir.mkdir(exist_ok=True)
    
    print("="*70)
    print("🎬 FINAL VIDEO VERIFICATION")
    print("="*70)
    print(f"Time: {datetime.now().strftime('%H:%M:%S')}")
    print(f"Video: {video_path}")
    print(f"Size: {video_path.stat().st_size / (1024*1024):.1f} MB\n")
    
    # Extract frames at key moments
    test_points = [
        (11.5, "Phrase 1: 'very beginning'"),
        (23.0, "Phrase 2: 'Do Re Mi'"),
        (47.5, "Cartoon 1: happy_female_deer"),
        (51.5, "Cartoon 2: smiling_drop_of_sun"),
        (25.0, "Karaoke: 'Do, Re, Mi' (if present)"),
        (45.0, "Karaoke: 'Do, a deer' (if present)")
    ]
    
    print("Extracting frames at key moments:")
    print("-"*50)
    
    for time, description in test_points:
        frame_name = f"frame_{time:.1f}s.png"
        frame_path = output_dir / frame_name
        
        cmd = [
            'ffmpeg', '-ss', str(time),
            '-i', str(video_path),
            '-frames:v', '1',
            '-y', str(frame_path)
        ]
        
        result = subprocess.run(cmd, capture_output=True)
        if result.returncode == 0:
            print(f"✓ {time:5.1f}s: {description}")
            print(f"         → {frame_name}")
        else:
            print(f"✗ {time:5.1f}s: Failed to extract")
    
    print("\n" + "="*70)
    print("✅ VERIFICATION COMPLETE")
    print("="*70)
    print(f"\n📁 Check frames in: {output_dir}")
    print("\nExpected to see:")
    print("  • White text 'very beginning' at 11.5s (top-right)")
    print("  • Yellow text 'Do Re Mi' at 23s (top-left)")
    print("  • Spring cartoon hopping at 47.5s")
    print("  • Spring cartoon floating at 51.5s")
    print("  • Karaoke captions at bottom (if karaoke worked)")
    
    # Also create a montage for easy viewing
    print("\nCreating montage of all frames...")
    montage_cmd = [
        'ffmpeg', '-pattern_type', 'glob',
        '-i', str(output_dir / '*.png'),
        '-filter_complex', 'tile=3x2',
        '-y', str(output_dir / 'montage.png')
    ]
    
    result = subprocess.run(montage_cmd, capture_output=True)
    if result.returncode == 0:
        print(f"✓ Montage created: {output_dir / 'montage.png'}")
    
    print("\n🎯 TO VERIFY:")
    print(f"1. Open video: open {video_path}")
    print(f"2. Check montage: open {output_dir / 'montage.png'}")
    print(f"3. Or check individual frames in: {output_dir}")

if __name__ == "__main__":
    extract_and_verify()