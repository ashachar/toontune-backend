#!/usr/bin/env python3
"""
Local test script for Lambda function.
Tests the animation processing without AWS deployment.
"""

import sys
import os
import json

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lambda_function.lambda_handler import process_video


def test_local():
    """Test the animation processing locally."""
    
    # Use the test video we've been working with
    input_video = "../uploads/assets/videos/do_re_mi.mov"
    output_video = "test_lambda_output.mp4"
    test_text = "HELLO"
    
    if not os.path.exists(input_video):
        print(f"❌ Test video not found: {input_video}")
        print("Please ensure the test video exists at the specified path")
        return False
    
    print(f"🎬 Testing animation with text: '{test_text}'")
    print(f"   Input: {input_video}")
    print(f"   Output: {output_video}")
    
    try:
        # Process the video
        process_video(input_video, test_text, output_video)
        
        if os.path.exists(output_video):
            file_size = os.path.getsize(output_video) / (1024 * 1024)  # MB
            print(f"✅ Success! Output video created: {output_video} ({file_size:.2f} MB)")
            
            # Convert to GIF for easy viewing
            gif_output = output_video.replace('.mp4', '.gif')
            os.system(f"ffmpeg -i {output_video} -vf 'fps=10,scale=400:-1' {gif_output} -y")
            if os.path.exists(gif_output):
                print(f"📸 GIF preview created: {gif_output}")
            
            return True
        else:
            print("❌ Output video was not created")
            return False
            
    except Exception as e:
        print(f"❌ Error during processing: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_local()