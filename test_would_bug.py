#!/usr/bin/env python3
"""
Test script to debug the "Would" word fog issue in word-level pipeline
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pipelines.word_level_pipeline.main import create_word_level_video

def test_would_bug():
    """Test the word-level pipeline with ai_math1.mp4 to debug the 'Would' issue"""
    
    print("=" * 80)
    print("TESTING WORD-LEVEL PIPELINE - 'Would' FOG BUG")
    print("=" * 80)
    print()
    print("This test will:")
    print("1. Process ai_math1.mp4 with the word-level pipeline")
    print("2. Show detailed debug output for the word 'Would'")
    print("3. Track scene assignments and fog transitions")
    print()
    print("Expected fix: 'Would' should appear in Scene 1, not be affected by Scene 0's fog")
    print("-" * 80)
    
    # Check if input video exists
    input_video = "uploads/assets/videos/ai_math1.mp4"
    if not os.path.exists(input_video):
        print(f"❌ Error: Input video not found at {input_video}")
        print("   Please ensure ai_math1.mp4 is in the uploads/assets/videos/ directory")
        return
    
    # Run the pipeline with debug output
    output_path = create_word_level_video(
        input_video_path=input_video,
        duration_seconds=6.0,  # Process 6 second segment
        output_name="ai_math1_would_bug_fixed.mp4"
    )
    
    if output_path:
        print("\n" + "=" * 80)
        print("✅ TEST COMPLETE")
        print(f"   Output video: {output_path}")
        print()
        print("Check the debug output above for:")
        print("  • Scene assignments for 'Would' and related words")
        print("  • Fog transition timing")
        print("  • Whether 'Would' is correctly assigned to Scene 1")
        print("=" * 80)
    else:
        print("\n❌ Pipeline failed - check debug output above")

if __name__ == "__main__":
    test_would_bug()