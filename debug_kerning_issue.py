#!/usr/bin/env python3
"""
Debug kerning issue between full word rendering vs individual letter rendering.
"""

import os
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2

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


def extract_letter_region(frame, x_start_pct=0.52, x_end_pct=0.65, y_start_pct=0.40, y_end_pct=0.50):
    """Extract the region where the last letter should be."""
    h, w = frame.shape[:2]
    x1 = int(w * x_start_pct)
    x2 = int(w * x_end_pct)
    y1 = int(h * y_start_pct)
    y2 = int(h * y_end_pct)
    return frame[y1:y2, x1:x2], (x1, y1)


def compare_rendering_methods():
    """Compare full word vs individual letter rendering."""
    
    # Setup
    width = 1168
    height = 526
    center_x = width // 2
    center_y = int(height * 0.45)
    font_size = 147
    text = "START"
    text_color = (255, 220, 0)
    
    print("Comparing rendering methods:")
    print(f"  Text: '{text}'")
    print(f"  Font size: {font_size}")
    print(f"  Center: ({center_x}, {center_y})")
    print()
    
    font = load_font(None, font_size)
    
    # Method 1: Full word rendering (like TextBehindSegment at scale 1.0)
    img1 = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    draw1 = ImageDraw.Draw(img1)
    
    # Calculate position for full word
    bbox = draw1.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    text_x = center_x - text_width // 2
    text_y = center_y - text_height // 2
    
    # Draw full word with effects
    # Shadow
    draw1.text((text_x + 3, text_y + 3), text, font=font, fill=(0, 0, 0, 100))
    # Outline
    for dx in [-2, 2]:
        for dy in [-2, 2]:
            draw1.text((text_x + dx, text_y + dy), text, font=font, fill=(255, 255, 255, 150))
    # Main text
    draw1.text((text_x, text_y), text, font=font, fill=(*text_color, 255))
    
    # Method 2: Individual letter rendering (like WordDissolve)
    img2 = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    draw2 = ImageDraw.Draw(img2)
    
    # Calculate base position for word
    full_bbox = draw2.textbbox((0, 0), text, font=font)
    full_width = full_bbox[2] - full_bbox[0]
    full_height = full_bbox[3] - full_bbox[1]
    base_x = center_x - full_width // 2
    baseline_y = center_y - full_height // 2
    
    # Draw each letter individually with kerning
    for i, letter in enumerate(text):
        # Calculate position using prefix width
        prefix = text[:i]
        if prefix:
            prefix_bbox = draw2.textbbox((0, 0), prefix, font=font)
            prefix_width = prefix_bbox[2] - prefix_bbox[0]
        else:
            prefix_width = 0
        
        letter_x = base_x + prefix_width
        letter_y = baseline_y
        
        # Draw with same effects
        # Shadow
        draw2.text((letter_x + 3, letter_y + 3), letter, font=font, fill=(0, 0, 0, 100))
        # Outline
        for dx in [-2, 2]:
            for dy in [-2, 2]:
                draw2.text((letter_x + dx, letter_y + dy), letter, font=font, fill=(255, 255, 255, 150))
        # Main letter
        draw2.text((letter_x, letter_y), letter, font=font, fill=(*text_color, 255))
        
        if i == len(text) - 1:  # Last letter
            print(f"  Last letter '{letter}' drawn at: ({letter_x}, {letter_y})")
    
    # Convert to numpy arrays
    arr1 = np.array(img1)[:, :, :3]  # RGB only
    arr2 = np.array(img2)[:, :, :3]
    
    # Extract last letter region
    region1, (x_off, y_off) = extract_letter_region(arr1)
    region2, _ = extract_letter_region(arr2)
    
    # Find center of mass for last letter
    def find_text_center(region):
        gray = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY)
        bright = gray > 100
        if np.any(bright):
            y_coords, x_coords = np.where(bright)
            return np.mean(x_coords), np.mean(y_coords)
        return None, None
    
    cx1, cy1 = find_text_center(region1)
    cx2, cy2 = find_text_center(region2)
    
    if cx1 is not None and cx2 is not None:
        print(f"\nLast letter region center of mass:")
        print(f"  Full word rendering: ({cx1 + x_off:.1f}, {cy1 + y_off:.1f})")
        print(f"  Individual letters: ({cx2 + x_off:.1f}, {cy2 + y_off:.1f})")
        print(f"  X difference: {(cx2 - cx1):.1f} pixels")
        print(f"  Y difference: {(cy2 - cy1):.1f} pixels")
    
    # Save comparison images
    comparison = np.hstack([arr1, arr2])
    cv2.imwrite("rendering_comparison.png", cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))
    print("\n✓ Saved comparison to rendering_comparison.png")
    
    # Calculate pixel difference
    diff = np.abs(arr1.astype(float) - arr2.astype(float))
    diff_img = diff.astype(np.uint8)
    cv2.imwrite("rendering_difference.png", cv2.cvtColor(diff_img, cv2.COLOR_RGB2BGR))
    
    total_diff = np.sum(diff > 10)
    print(f"Total pixels with difference > 10: {total_diff}")
    
    # Check if renderings are identical
    if total_diff == 0:
        print("✅ Renderings are IDENTICAL")
    else:
        print(f"⚠️ Renderings differ in {total_diff} pixels")
        
        # Find where the differences are
        diff_mask = np.any(diff > 10, axis=2)
        diff_coords = np.where(diff_mask)
        if len(diff_coords[0]) > 0:
            min_x = np.min(diff_coords[1])
            max_x = np.max(diff_coords[1])
            print(f"  Differences occur between x={min_x} and x={max_x}")


if __name__ == "__main__":
    compare_rendering_methods()