#!/usr/bin/env python3
"""
Simple test to verify text positioning between TextBehindSegment and WordDissolve.
No masks, no complex processing - just the text rendering.
"""

import os
import sys
import numpy as np
from PIL import Image
import imageio

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.animations.text_behind_segment import TextBehindSegment
from utils.animations.word_dissolve import WordDissolve


def test_simple_transition():
    """Test simple transition without masks."""
    
    # Simple parameters
    width = 1168
    height = 526
    center_position = (width // 2, int(height * 0.45))
    font_size = 147
    
    # Create blank frames
    blank_frame = np.zeros((height, width, 3), dtype=np.uint8)
    blank_frame[:] = (50, 50, 50)  # Dark gray background
    
    # Create dummy image files
    dummy_img = Image.new('RGB', (100, 100), (0, 0, 0))
    dummy_img.save('/tmp/dummy.png')
    
    # Create animations
    text_animator = TextBehindSegment(
        element_path="/tmp/dummy.png",
        background_path="/tmp/dummy.png",
        position=center_position,
        text="START",
        font_size=font_size,
        text_color=(255, 220, 0),
        start_scale=1.0,  # Start at scale 1.0 for simplicity
        end_scale=1.0,    # Stay at scale 1.0
        phase1_duration=0.5,
        phase2_duration=0.5,
        phase3_duration=1.0,  # 30 frames at 30fps
        center_position=center_position,
        fps=30
    )
    
    word_dissolver = WordDissolve(
        element_path="/tmp/dummy.png",
        background_path="/tmp/dummy.png",
        position=center_position,
        word="START",
        font_size=font_size,
        text_color=(255, 220, 0),
        stable_duration=0.17,  # Add stable period
        dissolve_duration=0.67,
        dissolve_stagger=0.33,
        float_distance=30,
        randomize_order=False,  # Dissolve in order for consistency
        maintain_kerning=True,
        center_position=center_position,
        fps=30
    )
    
    frames = []
    
    # Frame 29: Last frame of TextBehindSegment (stable phase)
    print("Rendering TextBehindSegment at frame 59 (stable phase)...")
    frame_tbs = text_animator.render_text_frame(blank_frame.copy(), 59, None)
    frames.append(frame_tbs)
    
    # Frame 30: First frame of WordDissolve (no letters dissolving yet)
    print("Rendering WordDissolve at frame 0 (no dissolve yet)...")
    frame_wd = word_dissolver.render_word_frame(blank_frame.copy(), 0, None)
    frames.append(frame_wd)
    
    # Calculate difference
    diff = np.abs(frame_tbs.astype(float) - frame_wd.astype(float))
    max_diff = np.max(diff)
    total_diff = np.sum(diff > 10)
    
    print(f"\nComparison:")
    print(f"  Maximum pixel difference: {max_diff}")
    print(f"  Total pixels with diff > 10: {total_diff}")
    
    if total_diff == 0:
        print("✅ PERFECT MATCH! Frames are identical.")
    else:
        print(f"⚠️ Frames differ in {total_diff} pixels")
        
        # Find center of mass of text in each frame
        def find_text_center(frame):
            # Convert to grayscale
            gray = np.dot(frame[...,:3], [0.2989, 0.5870, 0.1140])
            bright = gray > 100
            if np.any(bright):
                y_coords, x_coords = np.where(bright)
                return np.mean(x_coords), np.mean(y_coords)
            return None, None
        
        cx1, cy1 = find_text_center(frame_tbs)
        cx2, cy2 = find_text_center(frame_wd)
        
        if cx1 is not None and cx2 is not None:
            print(f"\nText center of mass:")
            print(f"  TextBehindSegment: ({cx1:.1f}, {cy1:.1f})")
            print(f"  WordDissolve: ({cx2:.1f}, {cy2:.1f})")
            print(f"  Difference: ({cx2-cx1:.1f}, {cy2-cy1:.1f}) pixels")
    
    # Save comparison
    comparison = np.hstack([frame_tbs, frame_wd])
    imageio.imwrite("simple_transition_test.png", comparison)
    print("\n✓ Saved comparison to simple_transition_test.png")
    
    # Also save difference image
    diff_img = (diff * 10).clip(0, 255).astype(np.uint8)  # Amplify differences
    imageio.imwrite("simple_transition_diff.png", diff_img)
    print("✓ Saved difference map to simple_transition_diff.png")


if __name__ == "__main__":
    test_simple_transition()