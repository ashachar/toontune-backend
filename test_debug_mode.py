#!/usr/bin/env python3
"""
Test script for debug mode rendering.
Sets IS_DEBUG_MODE=true and runs the pipeline to generate both regular and debug videos.
"""

import os
import subprocess
from pathlib import Path

def test_debug_mode():
    """Test the debug mode rendering"""
    
    # Set debug mode environment variable
    os.environ['IS_DEBUG_MODE'] = 'true'
    
    print("🐛 DEBUG MODE TEST")
    print("=" * 60)
    print("Testing debug video generation with mask overlay and bounding boxes")
    print()
    
    # Use the original video with transcript
    input_video = "uploads/assets/videos/ai_math1.mp4"
    
    # Check if video exists
    if not os.path.exists(input_video):
        print(f"❌ Video not found: {input_video}")
        return False
    
    # Run the word-level pipeline with debug mode (process only 6 seconds)
    print("\n🎬 Running word-level pipeline with DEBUG MODE enabled...")
    print("   Processing first 6 seconds of video...")
    cmd = [
        'python', '-m', 'pipelines.word_level_pipeline',
        input_video,
        '6'  # Process only 6 seconds
    ]
    
    # Set environment variable for subprocess
    env = os.environ.copy()
    env['IS_DEBUG_MODE'] = 'true'
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False, env=env)
        print(result.stdout)
        if result.stderr:
            print("Debug info:", result.stderr)
        
        # Check for output files
        regular_output = "outputs/ai_math1_word_level_h264.mp4"
        debug_output = "outputs/ai_math1_word_level_h264_debug.mp4"
        
        print("\n📦 Output files:")
        if os.path.exists(regular_output):
            print(f"  ✅ Regular video: {regular_output}")
        else:
            print(f"  ❌ Regular video not found: {regular_output}")
        
        if os.path.exists(debug_output):
            print(f"  ✅ Debug video: {debug_output}")
            print("\n🔍 Debug video features:")
            print("  - Top half: Original video with text overlay")
            print("  - Bottom left: Binary foreground mask (white=person)")
            print("  - Bottom right: Mask overlay on frame")
            print("  - Green boxes: Front text")
            print("  - Blue boxes: Behind text")
            print("  - Text labels show word and [B] for behind")
            print("  - Timing info shows start time for each word")
        else:
            print(f"  ⚠️  Debug video not created (check if IS_DEBUG_MODE was set)")
        
        return os.path.exists(debug_output)
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running pipeline: {e}")
        if e.stdout:
            print("Output:", e.stdout)
        if e.stderr:
            print("Error output:", e.stderr)
        return False

if __name__ == "__main__":
    # First check if .env file exists and has IS_DEBUG_MODE
    env_path = Path(".env")
    if not env_path.exists():
        print("📝 Creating .env file with IS_DEBUG_MODE=true")
        with open(".env", "w") as f:
            f.write("IS_DEBUG_MODE=true\n")
    else:
        # Check if IS_DEBUG_MODE is in .env
        with open(".env", "r") as f:
            content = f.read()
        
        if "IS_DEBUG_MODE" not in content:
            print("📝 Adding IS_DEBUG_MODE=true to .env")
            with open(".env", "a") as f:
                f.write("\nIS_DEBUG_MODE=true\n")
        else:
            print("✅ IS_DEBUG_MODE already in .env")
    
    success = test_debug_mode()
    
    if success:
        print("\n🎉 Debug mode test completed successfully!")
        print("Check the debug video to verify mask overlay and bounding boxes")
    else:
        print("\n⚠️  Debug mode test completed but debug video was not created")
        print("Make sure IS_DEBUG_MODE=true is set in .env file")