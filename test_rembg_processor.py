#!/usr/bin/env python3
"""
Test script for the rembg-enabled text animation processor
"""

import sys
import os

# Add paths
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lambda', 'python'))

# Import the processor directly
import text_animation_processor
process_video = text_animation_processor.process_video

def test_processor():
    # Use one of the existing videos
    input_video = "lambda_output_v4.mp4"
    output_video = "test_rembg_output.mp4"
    text = "TEST"
    
    if not os.path.exists(input_video):
        print(f"[LAMBDA_ANIM] Error: Input video {input_video} not found")
        return False
    
    print(f"[LAMBDA_ANIM] Testing processor with rembg...")
    print(f"[LAMBDA_ANIM] Input: {input_video}")
    print(f"[LAMBDA_ANIM] Text: {text}")
    print(f"[LAMBDA_ANIM] Output: {output_video}")
    
    try:
        # Set environment variable for font size
        os.environ['TEXT_FONT_REL'] = '0.26'
        
        # Process the video
        process_video(input_video, text, output_video)
        
        # Check output exists
        if os.path.exists(output_video):
            size_mb = os.path.getsize(output_video) / (1024 * 1024)
            print(f"[LAMBDA_ANIM] ✓ Test successful! Output: {size_mb:.2f} MB")
            return True
        else:
            print(f"[LAMBDA_ANIM] Error: Output file not created")
            return False
            
    except Exception as e:
        print(f"[LAMBDA_ANIM] Error during processing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_processor()
    sys.exit(0 if success else 1)