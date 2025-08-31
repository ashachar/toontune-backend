#!/usr/bin/env python3
"""Test font size adjustment for text that would overflow screen boundaries."""

from utils.text_placement.two_position_layout import TwoPositionLayoutManager
from PIL import Image, ImageDraw, ImageFont
import numpy as np

def test_font_adjustment():
    """Test that font size automatically adjusts for long text."""
    
    # Create a test frame
    width, height = 1280, 720
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:] = (50, 50, 50)  # Dark gray background
    
    # Create layout manager
    layout = TwoPositionLayoutManager(width, height)
    
    # Test cases with different text lengths
    test_cases = [
        ("Short text", "Expected: normal size"),
        ("This is a medium length sentence that fits", "Expected: normal size"),
        ("This is a very long sentence that would definitely overflow the screen boundaries without automatic adjustment", "Expected: reduced size"),
        ("This is an extremely long sentence with many words that would absolutely positively definitely overflow even a very wide screen without proper automatic font size adjustment algorithm", "Expected: minimum size")
    ]
    
    print("Testing Font Size Adjustment")
    print("=" * 60)
    print(f"Screen dimensions: {width}x{height}")
    print(f"Safe width (85%): {int(width * 0.85)}px")
    print()
    
    for text, expected in test_cases:
        # Get font size for bottom position
        base_size = 65  # Max size for bottom
        adjusted_size = layout._calculate_safe_font_size(text, base_size)
        
        # Calculate actual text width
        font = ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc', adjusted_size)
        img = Image.new('RGB', (width, height))
        draw = ImageDraw.Draw(img)
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0] + 6  # Include outline
        
        print(f"Text: '{text[:50]}{'...' if len(text) > 50 else ''}'")
        print(f"  Length: {len(text)} chars")
        print(f"  Base font size: {base_size}px")
        print(f"  Adjusted size: {adjusted_size}px")
        print(f"  Reduction: {100 - (adjusted_size/base_size*100):.1f}%")
        print(f"  Text width: {text_width}px")
        print(f"  Fits in safe zone: {'✓' if text_width <= width * 0.85 else '✗'}")
        print(f"  {expected}")
        print()
    
    # Visual test - create an image showing the adjustments
    test_image = Image.new('RGB', (width, height * len(test_cases)), (50, 50, 50))
    draw = ImageDraw.Draw(test_image)
    
    for i, (text, _) in enumerate(test_cases):
        y_offset = i * height
        y_pos = y_offset + int(height * 0.85)
        
        # Get adjusted font size
        base_size = 65
        adjusted_size = layout._calculate_safe_font_size(text, base_size)
        font = ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc', adjusted_size)
        
        # Draw safe zone boundaries
        safe_start = int(width * 0.075)
        safe_end = int(width * 0.925)
        draw.line([(safe_start, y_offset), (safe_start, y_offset + height)], fill=(100, 100, 100), width=2)
        draw.line([(safe_end, y_offset), (safe_end, y_offset + height)], fill=(100, 100, 100), width=2)
        
        # Draw the text centered
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        x_pos = (width - text_width) // 2
        
        # Draw with outline effect
        for dx in range(-3, 4):
            for dy in range(-3, 4):
                if abs(dx) == 3 or abs(dy) == 3:
                    draw.text((x_pos + dx, y_pos + dy), text, font=font, fill=(255, 255, 255))
        
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                if abs(dx) == 2 or abs(dy) == 2:
                    draw.text((x_pos + dx, y_pos + dy), text, font=font, fill=(0, 0, 0))
        
        draw.text((x_pos, y_pos), text, font=font, fill=(255, 200, 0))
        
        # Add info text
        info_font = ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc', 20)
        info_text = f"Font size: {adjusted_size}px (reduced {100 - (adjusted_size/base_size*100):.0f}%)"
        draw.text((10, y_offset + 10), info_text, font=info_font, fill=(200, 200, 200))
    
    # Save test image
    test_image.save('outputs/font_adjustment_test.png')
    print("Visual test saved to: outputs/font_adjustment_test.png")
    print("\nDark gray vertical lines show the safe zone boundaries (85% of width)")

if __name__ == "__main__":
    test_font_adjustment()