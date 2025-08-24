#!/usr/bin/env python3
"""
Test script to verify video frame freezing works correctly
"""

import subprocess
from pathlib import Path

def test_freeze_frame():
    """Test creating a video with frozen frame at 4s for 2s"""
    
    input_video = "/Users/amirshachar/Desktop/Amir/Projects/Personal/toontune/backend/uploads/assets/videos/ai_math.mp4"
    output_video = "/tmp/test_frozen.mp4"
    
    # Create a video that freezes at 4s for 2s using setpts filter
    # This manipulates the presentation timestamps to freeze the frame
    
    filter_complex = """
    [0:v]trim=0:4,setpts=PTS-STARTPTS[v1];
    [0:v]trim=4:4.001,setpts=PTS-STARTPTS,loop=loop=60:size=1,setpts=N/30/TB,trim=0:2[frozen];
    [0:v]trim=4:60,setpts=PTS-STARTPTS+2[v2];
    [v1][frozen][v2]concat=n=3:v=1:a=0[outv]
    """
    
    cmd = [
        "ffmpeg", "-y",
        "-i", input_video,
        "-filter_complex", filter_complex,
        "-map", "[outv]",
        "-c:v", "libx264", "-preset", "fast",
        output_video
    ]
    
    print("Creating test video with frozen frame...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"✅ Success! Created: {output_video}")
        print("\nNow checking if frame is actually frozen...")
        
        # Extract frames to verify
        subprocess.run([
            "ffmpeg", "-y",
            "-i", output_video,
            "-ss", "3.9", "-t", "3",
            "-r", "10",
            "/tmp/verify_%03d.png"
        ], capture_output=True)
        
        # Check frames
        from PIL import Image
        import numpy as np
        
        frames = []
        for i in range(1, 31):
            try:
                img = Image.open(f'/tmp/verify_{i:03d}.png')
                frames.append(np.array(img))
            except:
                break
        
        print(f"\nAnalyzing {len(frames)} frames:")
        frozen_count = 0
        for i in range(1, len(frames)):
            diff = np.sum(np.abs(frames[i] - frames[i-1]))
            timestamp = 3.9 + (i * 0.1)
            is_frozen = diff < 10000
            if is_frozen:
                frozen_count += 1
                print(f"  Frame {i:02d} @ {timestamp:.1f}s: FROZEN!")
        
        if frozen_count > 15:  # Should have ~20 frozen frames for 2s @ 10fps
            print(f"\n✅ SUCCESS! Video freezes correctly ({frozen_count} frozen frames)")
        else:
            print(f"\n❌ FAILED! Only {frozen_count} frozen frames detected")
            
    else:
        print(f"❌ FFmpeg failed: {result.stderr[:500]}")

if __name__ == "__main__":
    test_freeze_frame()