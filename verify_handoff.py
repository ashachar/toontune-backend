#!/usr/bin/env python3
"""
Quick verification that the handoff data is working correctly.
"""

import os
import sys
import numpy as np
from PIL import Image

# Add paths
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.animations.text_behind_segment import TextBehindSegment
from utils.animations.word_dissolve import WordDissolve


def test_handoff_data():
    """Test that handoff data preserves letter positions."""
    
    # Create a simple test frame
    test_frame = np.zeros((400, 800, 3), dtype=np.uint8)
    test_frame.fill(50)  # Dark gray background
    
    center_position = (400, 200)
    font_size = 100
    
    # Create TextBehindSegment
    video_path = "/Users/amirshachar/Desktop/Amir/Projects/Personal/toontune/backend/uploads/assets/videos/do_re_mi.mov"
    text_animator = TextBehindSegment(
        element_path=video_path,
        background_path=video_path, 
        position=center_position,
        text="START",
        font_size=font_size,
        center_position=center_position,
        fps=30
    )
    
    # Render final frame to populate handoff data
    final_frame = text_animator.render_text_frame(test_frame, 89)  # Last frame of phase 3
    
    # Get handoff data
    handoff_data = text_animator.get_handoff_data()
    
    print("🔍 Handoff Data Verification:")
    print(f"  - Text: {handoff_data.get('text')}")
    print(f"  - Final font size: {handoff_data.get('final_font_size')}")
    print(f"  - Final center position: {handoff_data.get('final_center_position')}")
    print(f"  - Number of letter positions: {len(handoff_data.get('final_letter_positions', []))}")
    
    if handoff_data.get('final_letter_positions'):
        print("  - Letter positions:")
        for i, (x, y, letter) in enumerate(handoff_data['final_letter_positions']):
            print(f"    {i}: '{letter}' at ({x}, {y})")
    
    # Create WordDissolve with handoff data
    word_dissolver = WordDissolve(
        element_path=video_path,
        background_path=video_path,
        position=center_position,
        word="START",
        font_size=font_size,
        handoff_data=handoff_data,
        fps=30
    )
    
    # Test that WordDissolve uses the frozen positions
    calculated_positions = word_dissolver.calculate_letter_positions()
    
    print(f"\n✅ WordDissolve Position Verification:")
    print(f"  - Using frozen positions: {word_dissolver.frozen_letter_positions is not None}")
    print(f"  - Using frozen font size: {word_dissolver.frozen_font_size}")
    print(f"  - Calculated positions match handoff:")
    
    if handoff_data.get('final_letter_positions'):
        for i, ((calc_x, calc_y), (orig_x, orig_y, letter)) in enumerate(zip(calculated_positions, handoff_data['final_letter_positions'])):
            match = calc_x == orig_x and calc_y == orig_y
            status = "✓" if match else "✗"
            print(f"    {status} Letter '{letter}': calc=({calc_x},{calc_y}) vs orig=({orig_x},{orig_y})")
    
    # Test rendering a frame
    dissolve_frame = word_dissolver.render_word_frame(test_frame, 0)
    
    print(f"\n🎬 Animation Test:")
    print(f"  - Frame rendered successfully: {dissolve_frame is not None}")
    print(f"  - Frame shape: {dissolve_frame.shape}")
    
    print("\n🎯 Test Results:")
    if handoff_data.get('final_letter_positions') and len(calculated_positions) == len(handoff_data['final_letter_positions']):
        all_match = all(
            calc_x == orig_x and calc_y == orig_y 
            for (calc_x, calc_y), (orig_x, orig_y, _) in zip(calculated_positions, handoff_data['final_letter_positions'])
        )
        if all_match:
            print("  ✅ SUCCESS: All letter positions preserved during handoff!")
        else:
            print("  ❌ FAILURE: Letter positions changed during handoff")
    else:
        print("  ❌ FAILURE: Handoff data incomplete or missing")
    
    return handoff_data, calculated_positions


if __name__ == "__main__":
    test_handoff_data()