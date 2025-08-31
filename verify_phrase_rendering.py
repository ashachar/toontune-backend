#!/usr/bin/env python3
"""
Verify that phrase-level rendering is working by extracting key frames
"""

import subprocess
import os

def extract_frames():
    """Extract frames at key moments to verify rendering"""
    
    video = "outputs/ai_math1_word_level_h264.mp4"
    
    # Key moments to check
    times = [
        (2.0, "would_appear"),
        (3.4, "be_behind_head"),  # Critical moment - "be" should be behind
        (4.5, "if_visible")
    ]
    
    print("📸 Extracting frames to verify phrase rendering...")
    
    for time, name in times:
        output = f"outputs/verify_{name}.png"
        cmd = [
            'ffmpeg', '-y',
            '-ss', str(time),
            '-i', video,
            '-frames:v', '1',
            output
        ]
        subprocess.run(cmd, capture_output=True)
        print(f"   Frame at {time}s saved to: {output}")
    
    print("\n✅ Frames extracted! Check these images:")
    print("   - outputs/verify_would_appear.png (2.0s) - 'Would' starting to appear")
    print("   - outputs/verify_be_behind_head.png (3.4s) - 'be' should be behind head")
    print("   - outputs/verify_if_visible.png (4.5s) - 'if' should be visible on sides")
    print("\n🔍 What to look for:")
    print("   1. Text should be DARK BLUE color (not white)")
    print("   2. Text should be visible on background areas")
    print("   3. Text should be hidden where it overlaps with person")
    print("   4. Entire phrase 'Would you be surprised if' should be rendered together")

if __name__ == "__main__":
    extract_frames()