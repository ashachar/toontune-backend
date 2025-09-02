#!/usr/bin/env python3
"""
Test script to verify disappearance animations are working correctly.
Creates a test video with ASS captions to ensure text animates out properly.
"""

import os
import sys

# Add the parent directory to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sam2_head_aware_sandwich import main

def test_disappearance():
    """Run the main script and verify disappearance animations."""
    print("=" * 60)
    print("TESTING DISAPPEARANCE ANIMATIONS")
    print("=" * 60)
    print("\nThis test will process the AI math video with ASS captions")
    print("and verify that disappearance animations are working.\n")
    print("Watch for these log messages:")
    print("  - '🎬 DISAPPEAR START' - Animation beginning")
    print("  - '🎬 DISAPPEAR MID' - Halfway through animation")
    print("  - '🎬 DISAPPEAR END' - Animation complete")
    print("  - '🎬 HEAD-TRACK DISAPPEAR' - Head-tracking phrase disappearing")
    print("\n" + "=" * 60 + "\n")
    
    # Run the main function
    main()
    
    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)
    print("\nPlease check the output video at:")
    print("  ../../outputs/ai_math1_ass_sandwich.mp4")
    print("\nVerify that:")
    print("1. Text fades/slides out smoothly at scene ends (not abrupt)")
    print("2. 'Yes,' animates out with other Scene 0 phrases")
    print("3. All phrases in a scene use the same disappearance effect")
    print("4. Text behind head stays behind during disappearance")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    test_disappearance()