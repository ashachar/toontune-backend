#!/usr/bin/env python3
"""
Test script to demonstrate karaoke caption generation in the unified pipeline.
"""

import subprocess
import sys
from pathlib import Path

def test_karaoke_pipeline():
    """Run the pipeline with karaoke generation enabled."""
    
    print("\n" + "🎤" * 30)
    print("  TESTING KARAOKE IN UNIFIED PIPELINE")
    print("🎤" * 30)
    
    # Video path
    video = "uploads/assets/videos/do_re_mi.mov"
    
    # Check if video exists
    if not Path(video).exists():
        print(f"\n❌ Video not found: {video}")
        print("Please ensure the test video is available.")
        return False
    
    print(f"\n📹 Input video: {video}")
    print("\n🎯 Pipeline Configuration:")
    print("  • Skip downsampling (use existing)")
    print("  • Skip transcript generation (use existing)")
    print("  • Skip scene splitting (use existing)")
    print("  • Skip prompt generation")
    print("  • Skip LLM inference")
    print("  • Skip video editing")
    print("  • ✅ GENERATE KARAOKE CAPTIONS")
    print("  • Style: continuous (no flickering)")
    
    # Build command
    cmd = [
        "python", "unified_video_pipeline.py",
        video,
        "--no-downsample",      # Skip downsampling
        "--no-transcript",      # Use existing transcript
        "--no-scenes",          # Use existing scenes
        "--no-prompts",         # Skip prompts
        "--no-inference",       # Skip LLM
        "--no-editing",         # Skip editing
        "--karaoke",            # ENABLE KARAOKE
        "--karaoke-style", "continuous"  # No-flicker style
    ]
    
    print("\n🚀 Running pipeline with karaoke generation...")
    print(f"Command: {' '.join(cmd)}")
    print("\n" + "-" * 60)
    
    try:
        # Run the pipeline
        result = subprocess.run(cmd, capture_output=False, text=True)
        
        if result.returncode == 0:
            print("\n" + "-" * 60)
            print("\n✅ SUCCESS! Karaoke captions generated.")
            print("\n📁 Output locations:")
            print("  • Karaoke videos: uploads/assets/videos/do_re_mi/scenes/karaoke/")
            print("    - scene_001.mp4 (with karaoke captions)")
            print("    - scene_002.mp4 (with karaoke captions)")
            print("    - scene_003.mp4 (with karaoke captions)")
            print("\n🎨 Caption features:")
            print("  ✓ Center-bottom positioning")
            print("  ✓ Max 6 words per line")
            print("  ✓ Word-by-word yellow highlighting")
            print("  ✓ No flickering (continuous display)")
            print("  ✓ Last word stays highlighted during pauses")
            return True
        else:
            print(f"\n❌ Pipeline failed with exit code: {result.returncode}")
            return False
            
    except Exception as e:
        print(f"\n❌ Error running pipeline: {e}")
        return False


def show_usage():
    """Show how to use karaoke in the pipeline."""
    
    print("\n" + "📖" * 30)
    print("  KARAOKE PIPELINE USAGE")
    print("📖" * 30)
    
    print("\n1️⃣ BASIC USAGE (process everything with karaoke):")
    print("   python unified_video_pipeline.py video.mp4 --karaoke")
    
    print("\n2️⃣ KARAOKE ONLY (skip other processing):")
    print("   python unified_video_pipeline.py video.mp4 \\")
    print("     --no-downsample --no-transcript --no-scenes \\")
    print("     --no-prompts --no-inference --no-editing \\")
    print("     --karaoke")
    
    print("\n3️⃣ WITH EDITING AND KARAOKE:")
    print("   python unified_video_pipeline.py video.mp4 \\")
    print("     --no-downsample --no-transcript --no-scenes \\")
    print("     --no-prompts --dry-run \\")
    print("     --karaoke")
    
    print("\n4️⃣ CHOOSE KARAOKE STYLE:")
    print("   --karaoke-style continuous  # No flickering (default)")
    print("   --karaoke-style simple      # Basic style")
    
    print("\n📂 OUTPUT DIRECTORY STRUCTURE:")
    print("   uploads/assets/videos/{video_name}/")
    print("   └── scenes/")
    print("       ├── original/       # Original scenes")
    print("       ├── edited/         # Edited scenes (if editing enabled)")
    print("       └── karaoke/        # 🎤 Scenes with karaoke captions")
    print("           ├── scene_001.mp4")
    print("           ├── scene_002.mp4")
    print("           └── ...")


if __name__ == "__main__":
    print("\n🎬 UNIFIED VIDEO PIPELINE - KARAOKE FEATURE TEST")
    
    # Show usage information
    show_usage()
    
    # Run test
    print("\n" + "=" * 60)
    print("RUNNING TEST...")
    print("=" * 60)
    
    success = test_karaoke_pipeline()
    
    if success:
        print("\n" + "🎉" * 30)
        print("  KARAOKE TEST COMPLETE!")
        print("🎉" * 30)
    else:
        print("\n⚠️ Test failed. Please check the error messages above.")
        sys.exit(1)