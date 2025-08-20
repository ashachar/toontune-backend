#!/usr/bin/env python3
"""
Debug the exact positioning at the transition between TextBehindSegment and WordDissolve.
"""

import os
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Add paths
sys.path.insert(0, os.path.expanduser("~/sam2"))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.animations.text_behind_segment import TextBehindSegment
from utils.animations.word_dissolve import WordDissolve


def load_font(font_path, size):
    """Load font with specified size."""
    system_fonts = [
        "/System/Library/Fonts/Helvetica.ttc",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "arial.ttf"
    ]
    for font in system_fonts:
        if os.path.exists(font):
            return ImageFont.truetype(font, size)
    return ImageFont.load_default()


def test_positioning():
    """Test exact positioning calculation at transition."""
    
    # Test parameters matching the actual animation
    width = 1168
    height = 526
    center_position = (width // 2, int(height * 0.45))
    font_size = int(min(150, height * 0.28))
    text = "START"
    
    print(f"Test Configuration:")
    print(f"  Canvas: {width}x{height}")
    print(f"  Center position: {center_position}")
    print(f"  Font size: {font_size}")
    print(f"  Text: '{text}'")
    print()
    
    # Create test image
    test_img = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(test_img)
    font = load_font(None, font_size)
    
    center_x, center_y = center_position
    
    # ========== TextBehindSegment Calculation (at scale 1.0) ==========
    print("TextBehindSegment Position Calculation (scale=1.0):")
    
    # Get base text dimensions
    base_bbox = draw.textbbox((0, 0), text, font=font)
    base_width = base_bbox[2] - base_bbox[0]
    base_height = base_bbox[3] - base_bbox[1]
    
    # Calculate base position
    base_x = center_x - base_width // 2
    base_y = center_y - base_height // 2
    
    # At scale 1.0, scaled dimensions equal base dimensions
    current_scale = 1.0
    scaled_width = base_width
    scaled_height = base_height
    
    # Center the scaled text on the base position
    text_x = base_x - (scaled_width - base_width) // 2
    text_y = base_y - (scaled_height - base_height) // 2
    
    print(f"  Base bbox: {base_bbox}")
    print(f"  Base dimensions: {base_width}x{base_height}")
    print(f"  Base position: ({base_x}, {base_y})")
    print(f"  Final text position: ({text_x}, {text_y})")
    print(f"  Last letter 'T' approx at: x={text_x + base_width - 20}")
    print()
    
    # ========== WordDissolve Calculation ==========
    print("WordDissolve Position Calculation:")
    
    # Get full word dimensions
    full_bbox = draw.textbbox((0, 0), text, font=font)
    full_width = full_bbox[2] - full_bbox[0]
    full_height = full_bbox[3] - full_bbox[1]
    
    # Calculate baseline Y
    baseline_y = center_y - full_height // 2
    
    # Calculate base X for word
    word_base_x = center_x - full_width // 2
    
    print(f"  Full bbox: {full_bbox}")
    print(f"  Full dimensions: {full_width}x{full_height}")
    print(f"  Word base X: {word_base_x}")
    print(f"  Baseline Y: {baseline_y}")
    
    # Calculate position for last letter 'T'
    prefix = text[:-1]  # "STAR"
    prefix_bbox = draw.textbbox((0, 0), prefix, font=font)
    prefix_width = prefix_bbox[2] - prefix_bbox[0]
    last_letter_x = word_base_x + prefix_width
    
    print(f"  Prefix '{prefix}' width: {prefix_width}")
    print(f"  Last letter 'T' position: ({last_letter_x}, {baseline_y})")
    print()
    
    # ========== Comparison ==========
    print("Position Comparison:")
    print(f"  TextBehindSegment text starts at: ({text_x}, {text_y})")
    print(f"  WordDissolve text starts at: ({word_base_x}, {baseline_y})")
    print(f"  X difference: {word_base_x - text_x}")
    print(f"  Y difference: {baseline_y - text_y}")
    
    # Check if they're the same
    if text_x == word_base_x and text_y == baseline_y:
        print("  ✅ Positions MATCH!")
    else:
        print("  ❌ Positions DO NOT MATCH!")
        
    # Calculate last letter positions
    tbs_last_x = text_x + base_width - 20  # Approximate
    wd_last_x = last_letter_x
    
    print()
    print(f"Last Letter 'T' Position:")
    print(f"  TextBehindSegment: x≈{tbs_last_x}")
    print(f"  WordDissolve: x={wd_last_x}")
    print(f"  Difference: {wd_last_x - tbs_last_x:.1f} pixels")
    
    # Test with actual bbox calculations
    print()
    print("Detailed bbox analysis:")
    
    # Get the exact position where 'T' would be drawn
    test_img2 = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    draw2 = ImageDraw.Draw(test_img2)
    
    # Draw at TextBehindSegment position
    draw2.text((text_x, text_y), text, font=font, fill=(255, 255, 255, 255))
    tbs_array = np.array(test_img2)
    tbs_white = np.where(tbs_array[:, :, 3] > 0)
    if len(tbs_white[1]) > 0:
        tbs_rightmost = np.max(tbs_white[1])
        print(f"  TextBehindSegment rightmost pixel: {tbs_rightmost}")
    
    # Draw at WordDissolve position
    test_img3 = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    draw3 = ImageDraw.Draw(test_img3)
    draw3.text((word_base_x, baseline_y), text, font=font, fill=(255, 255, 255, 255))
    wd_array = np.array(test_img3)
    wd_white = np.where(wd_array[:, :, 3] > 0)
    if len(wd_white[1]) > 0:
        wd_rightmost = np.max(wd_white[1])
        print(f"  WordDissolve rightmost pixel: {wd_rightmost}")
        
        if 'tbs_rightmost' in locals():
            print(f"  Rightmost pixel difference: {wd_rightmost - tbs_rightmost} pixels")


if __name__ == "__main__":
    test_positioning()