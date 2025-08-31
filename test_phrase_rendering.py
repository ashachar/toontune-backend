#!/usr/bin/env python3
"""
Test script for phrase-level rendering of behind text.
Tests the first 6 seconds where "Would you be surprised if" should appear behind the person.
"""

import os
import json
import subprocess
from pathlib import Path

def test_phrase_rendering():
    """Test rendering with the new phrase-level approach"""
    
    # Input video (first 6 seconds only)
    input_video = "uploads/assets/videos/ai_math1.mp4"
    output_video = "outputs/test_phrase_behind_6sec.mp4"
    
    # First extract just 6 seconds for testing
    temp_video = "outputs/ai_math1_6sec.mp4"
    print("📹 Extracting first 6 seconds for testing...")
    cmd = [
        'ffmpeg', '-y',
        '-i', input_video,
        '-t', '6',
        '-c:v', 'libx264',
        '-preset', 'fast',
        '-crf', '18',
        '-c:a', 'copy',
        temp_video
    ]
    subprocess.run(cmd, check=True)
    print(f"   Created: {temp_video}")
    
    # Now run the word-level pipeline on this short video
    print("\n🎬 Running word-level pipeline with phrase rendering...")
    cmd = [
        'python', '-m', 'pipelines.word_level_pipeline',
        temp_video,
        '6'  # Process 6 seconds
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(result.stdout)
        if result.stderr:
            print("Warnings/Info:", result.stderr)
        
        # The pipeline creates output with _word_level_h264.mp4 suffix
        actual_output = temp_video.replace('.mp4', '_word_level_h264.mp4')
        print(f"\n✅ Success! Output video created: {actual_output}")
        print("\n🔍 Check the output video to verify:")
        print("   1. 'Would you be surprised if' should appear behind the person's head")
        print("   2. Text should be visible on both sides of the head")
        print("   3. Text should be dark blue color")
        print("   4. Text should be hidden where it overlaps with the person")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running pipeline: {e}")
        if e.stdout:
            print("Output:", e.stdout)
        if e.stderr:
            print("Error output:", e.stderr)
        return False
    
    return True

if __name__ == "__main__":
    success = test_phrase_rendering()
    if success:
        print("\n🎉 Test completed successfully!")
    else:
        print("\n❌ Test failed - check the errors above")