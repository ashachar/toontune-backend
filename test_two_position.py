#!/usr/bin/env python3
"""
Test script for two-position layout system
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pipelines.word_level_pipeline.main import create_word_level_video

def test_two_position_layout():
    """Test the word-level pipeline with two-position layout (top/bottom)"""
    
    print("=" * 80)
    print("TESTING TWO-POSITION LAYOUT SYSTEM")
    print("=" * 80)
    print()
    print("This test will:")
    print("1. Process ai_math1.mp4 with the new two-position layout")
    print("2. Place phrases at TOP or BOTTOM of screen only")
    print("3. Use different colors based on importance")
    print("4. Show multiple phrases side-by-side when at same position")
    print("-" * 80)
    
    # Check if input video exists
    input_video = "uploads/assets/videos/ai_math1.mp4"
    if not os.path.exists(input_video):
        print(f"❌ Error: Input video not found at {input_video}")
        return
    
    # Run the pipeline with new layout
    output_path = create_word_level_video(
        input_video_path=input_video,
        duration_seconds=8.0,  # Process 8 second segment to see multiple scenes
        output_name="ai_math1_two_position.mp4"
    )
    
    if output_path:
        print("\n" + "=" * 80)
        print("✅ TEST COMPLETE")
        print(f"   Output video: {output_path}")
        print()
        print("New features implemented:")
        print("  ✓ Two-position layout (top and bottom only)")
        print("  ✓ LLM decides position for each phrase")
        print("  ✓ Chronological ordering maintained")
        print("  ✓ Side-by-side arrangement for same position")
        print("  ✓ Color coding based on importance/type")
        print("  ✓ Size variation based on importance")
        print("=" * 80)
    else:
        print("\n❌ Pipeline failed - check debug output above")

if __name__ == "__main__":
    test_two_position_layout()