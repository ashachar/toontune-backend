#!/usr/bin/env python3
"""
Test script to verify micro-entrance animations for individual words.
Tests Path 2 where phrases have word timings but use whole-phrase effects.
"""

import os
import sys

# Add the parent directory to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sam2_head_aware_sandwich import main

def test_micro_entrance():
    """Run the main script and verify micro-entrance animations."""
    print("=" * 60)
    print("TESTING MICRO-ENTRANCE ANIMATIONS (Path 2)")
    print("=" * 60)
    print("\nThis test will process the AI math video with ASS captions")
    print("and verify that individual words fade/slide in smoothly.\n")
    print("Expected behavior:")
    print("  - Each word fades in over 200ms when its timestamp arrives")
    print("  - Words slide in slightly from the same direction as phrase effect")
    print("  - No abrupt appearances or disappearing/reappearing")
    print("\nWatch for these log messages:")
    print("  - '💫 MICRO-ENTRANCE' - Individual word starting to fade in")
    print("  - '🎭 ENTRANCE' - Phrase-level entrance animations")
    print("  - '🎬 DISAPPEAR' - Disappearance animations")
    print("\n" + "=" * 60 + "\n")
    
    # Run the main function
    main()
    
    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)
    print("\nPlease check the output video at:")
    print("  ../../outputs/ai_math1_sam2_head_aware_h264.mp4")
    print("\nVerify that:")
    print("1. Individual words fade in smoothly over 200ms")
    print("2. 'AI created new math' words appear one by one, not all at once")
    print("3. Each word has a subtle slide effect matching the phrase direction")
    print("4. No words disappear and reappear")
    print("5. The overall phrase effect still works correctly")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    test_micro_entrance()