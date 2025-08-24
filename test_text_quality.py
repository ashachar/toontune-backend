#!/usr/bin/env python3
"""
Test script to verify text quality improvements.
Creates a simple test video and applies text animation with quality settings.
"""

import cv2
import numpy as np
import subprocess
import os
from pathlib import Path

def create_test_video(output_path="test_input.mp4", duration=5, fps=30, width=1920, height=1080):
    """Create a simple test video with gradient background."""
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    total_frames = duration * fps
    for i in range(total_frames):
        # Create gradient background
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        # Blue to purple gradient
        for y in range(height):
            color_val = int((y / height) * 100)
            frame[y, :] = [100 - color_val, 50, color_val + 100]
        
        # Add some moving element for reference
        circle_x = int(width * (0.3 + 0.4 * (i / total_frames)))
        cv2.circle(frame, (circle_x, height // 2), 50, (255, 255, 255), -1)
        
        out.write(frame)
    
    out.release()
    print(f"[TEXT_QUALITY] Created test video: {output_path}")
    return output_path

def test_quality_settings():
    """Test different quality settings and compare."""
    
    # Create test video
    test_video = create_test_video()
    
    # Test configurations
    tests = [
        {
            "name": "low_quality",
            "args": [
                "--text", "QUALITY TEST",
                "--supersample", "2",
                "--pixfmt", "yuv420p",
                "--crf", "28",
                "--preset", "ultrafast",
                "--debug"
            ]
        },
        {
            "name": "high_quality",
            "args": [
                "--text", "QUALITY TEST",
                "--supersample", "12",
                "--pixfmt", "yuv444p",
                "--crf", "18",
                "--preset", "slow",
                "--debug"
            ]
        },
        {
            "name": "rgb_quality",
            "args": [
                "--text", "QUALITY TEST",
                "--supersample", "12",
                "--pixfmt", "rgb24",
                "--crf", "16",
                "--preset", "slow",
                "--tune", "animation",
                "--debug"
            ]
        }
    ]
    
    # Try to find a font
    font_candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
        "/Library/Fonts/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]
    
    font_path = None
    for candidate in font_candidates:
        if os.path.isfile(candidate):
            font_path = candidate
            print(f"[TEXT_QUALITY] Found font: {font_path}")
            break
    
    # Run tests
    for test in tests:
        print(f"\n{'='*60}")
        print(f"[TEXT_QUALITY] Running test: {test['name']}")
        print(f"{'='*60}")
        
        output_name = f"test_output_{test['name']}.mp4"
        cmd = ["python", "apply_3d_text_animation.py", test_video] + test['args']
        
        # Add font if found
        if font_path:
            cmd.extend(["--font", font_path])
        
        # Set output name
        cmd.extend(["--output", output_name])
        
        print(f"[TEXT_QUALITY] Command: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            print(result.stdout)
            if result.stderr:
                print(f"[TEXT_QUALITY] Errors: {result.stderr}")
            
            # Check output video pixel format
            if os.path.exists(f"{Path(output_name).stem}_hq.mp4"):
                check_cmd = [
                    "ffprobe", "-v", "error",
                    "-select_streams", "v:0",
                    "-show_entries", "stream=pix_fmt",
                    "-of", "default=nw=1:nk=1",
                    f"{Path(output_name).stem}_hq.mp4"
                ]
                pixfmt_result = subprocess.run(check_cmd, capture_output=True, text=True)
                print(f"[TEXT_QUALITY] Output pixel format: {pixfmt_result.stdout.strip()}")
        except Exception as e:
            print(f"[TEXT_QUALITY] Test failed: {e}")
    
    print(f"\n{'='*60}")
    print("[TEXT_QUALITY] All tests complete!")
    print("Compare the output videos to see quality differences:")
    print("  - test_output_low_quality_hq.mp4 (baseline)")
    print("  - test_output_high_quality_hq.mp4 (recommended)")
    print("  - test_output_rgb_quality_hq.mp4 (maximum quality)")
    print("Look for:")
    print("  - Text edge smoothness (especially diagonal lines in 'A', 'V', etc.)")
    print("  - Color accuracy at edges")
    print("  - Overall text sharpness")

if __name__ == "__main__":
    test_quality_settings()