#!/usr/bin/env python3
"""
Test that the handoff between TextBehindSegment and WordDissolve is pixel-perfect.
"""

import os
import sys
import numpy as np
from PIL import Image, ImageDraw
import cv2

# Add paths
sys.path.insert(0, os.path.expanduser("~/sam2"))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.animations.text_behind_segment import TextBehindSegment
from utils.animations.word_dissolve import WordDissolve


def test_handoff():
    """Test exact handoff between animations."""
    
    # Setup
    width, height = 1168, 526
    center_position = (width // 2, int(height * 0.45))
    font_size = int(min(150, height * 0.28))
    
    print(f"Test Configuration:")
    print(f"  Canvas: {width}x{height}")
    print(f"  Center: {center_position}")
    print(f"  Font size: {font_size}")
    print()
    
    # Create blank frames
    blank_frame = np.ones((height, width, 3), dtype=np.uint8) * 50  # Dark gray
    
    # Phase timing
    phase1_frames = 30  # Shrinking
    phase2_frames = 20  # Moving behind
    phase3_frames = 40  # Stable behind
    total_tbs_frames = phase1_frames + phase2_frames + phase3_frames
    
    # Create TextBehindSegment animation
    video_path = "uploads/assets/videos/do_re_mi.mov"
    text_animator = TextBehindSegment(
        element_path=video_path,
        background_path=video_path,
        position=center_position,
        text="START",
        font_size=font_size,
        text_color=(255, 220, 0),
        start_scale=2.0,
        end_scale=1.0,
        phase1_duration=phase1_frames / 30,
        phase2_duration=phase2_frames / 30,
        phase3_duration=phase3_frames / 30,
        center_position=center_position,
        fps=30
    )
    
    # Render last TBS frame
    print("Rendering last TBS frame...")
    last_tbs_frame = text_animator.render_text_frame(blank_frame.copy(), total_tbs_frames - 1)
    
    # Get handoff data
    handoff_data = text_animator.get_handoff_data()
    print(f"Handoff data acquired:")
    print(f"  - Letter positions: {len(handoff_data.get('final_letter_positions', []))}")
    print(f"  - Font size: {handoff_data.get('final_font_size')}")
    print(f"  - Text origin: {handoff_data.get('final_text_origin')}")
    print(f"  - Scale: {handoff_data.get('scale')}")
    print()
    
    # Create WordDissolve with handoff
    word_dissolver = WordDissolve(
        element_path=video_path,
        background_path=video_path,
        position=center_position,
        word="START",
        font_size=font_size,
        text_color=(255, 220, 0),
        stable_duration=0.17,  # 5 frames at 30fps
        dissolve_duration=0.67,
        dissolve_stagger=0.33,
        float_distance=30,
        randomize_order=False,  # Keep sequential for testing
        maintain_kerning=True,
        center_position=center_position,
        handoff_data=handoff_data,
        fps=30
    )
    
    # Render first WD frame (should be identical to last TBS frame)
    print("Rendering first WD frame...")
    first_wd_frame = word_dissolver.render_word_frame(blank_frame.copy(), 0)
    
    # Compare frames
    print("\nComparing frames...")
    diff = np.abs(last_tbs_frame.astype(float) - first_wd_frame.astype(float))
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    
    print(f"  Max pixel difference: {max_diff}")
    print(f"  Mean pixel difference: {mean_diff:.2f}")
    
    # Find where differences occur
    if max_diff > 1:
        diff_mask = np.max(diff, axis=2) > 1
        diff_coords = np.where(diff_mask)
        print(f"  Pixels with differences: {len(diff_coords[0])}")
        
        # Save comparison image
        comparison = np.hstack([last_tbs_frame, first_wd_frame, (diff * 10).astype(np.uint8)])
        cv2.imwrite("handoff_comparison.png", cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))
        print("\n✓ Saved visual comparison to handoff_comparison.png")
        print("  (Left: Last TBS frame, Middle: First WD frame, Right: Difference x10)")
    
    # Test verdict
    print("\n" + "=" * 50)
    if max_diff <= 1:
        print("✅ HANDOFF IS PIXEL-PERFECT!")
    elif max_diff <= 5:
        print("⚠️ Minor differences detected (acceptable)")
    else:
        print("❌ Significant differences detected - handoff needs fixing")
    
    return max_diff <= 1


if __name__ == "__main__":
    success = test_handoff()