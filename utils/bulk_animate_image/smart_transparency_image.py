#!/usr/bin/env python3
"""
Smart Background Removal
Only removes background if image doesn't already have transparency
Detects and cleans up false negatives from rembg
"""

import numpy as np
from PIL import Image
from rembg import remove
import warnings
warnings.filterwarnings("ignore")

def has_transparency(img):
    """Check if image already has significant transparency"""
    if img.mode != 'RGBA':
        return False
    
    alpha = np.array(img)[:, :, 3]
    transparent_pixels = np.sum(alpha < 128)
    total_pixels = alpha.size
    transparency_ratio = transparent_pixels / total_pixels
    
    # If more than 1% of pixels are transparent, consider it has transparency
    return transparency_ratio > 0.01

def detect_removed_background_color(original_img, nobg_img):
    """Detect what color was predominantly removed as background"""
    orig_array = np.array(original_img.convert('RGBA'))
    nobg_array = np.array(nobg_img)
    
    # Find pixels that became transparent
    transparent_mask = nobg_array[:, :, 3] < 128
    
    if not np.any(transparent_mask):
        return None, 0
    
    # Get RGB values of pixels that were removed
    removed_pixels = orig_array[transparent_mask, :3]
    
    if len(removed_pixels) == 0:
        return None, 0
    
    # Find the most common background color
    bg_color = np.median(removed_pixels, axis=0).astype(int)
    
    # Calculate how uniform the background was
    std_dev = np.std(removed_pixels, axis=0).mean()
    
    return tuple(bg_color), std_dev

def clean_false_negatives(img, bg_color, std_dev, threshold=30):
    """Remove pixels that match the detected background color"""
    if bg_color is None:
        return img
    
    img_array = np.array(img)
    
    # For uniform backgrounds (low std_dev), be more aggressive
    if std_dev < 10:
        color_distance_threshold = threshold
    else:
        color_distance_threshold = threshold * 1.5
    
    # Calculate color distance for all pixels
    color_diff = np.abs(img_array[:, :, :3] - np.array(bg_color))
    color_distance = np.sum(color_diff, axis=2)
    
    # Find pixels close to background color that aren't already transparent
    false_negatives = (color_distance < color_distance_threshold) & (img_array[:, :, 3] > 128)
    
    # Make them transparent
    img_array[false_negatives, 3] = 0
    
    cleaned_pixels = np.sum(false_negatives)
    
    return Image.fromarray(img_array), cleaned_pixels

def smart_remove_background(image_path, alpha_matting=False):
    """Smart background removal with false negative cleanup"""
    print(f"Smart background removal for: {image_path}")
    
    # Load original image
    original_img = Image.open(image_path)
    
    # Check if already has transparency
    if has_transparency(original_img):
        transparency_ratio = np.sum(np.array(original_img.convert('RGBA'))[:, :, 3] < 128) / (original_img.width * original_img.height)
        print(f"  Image already has {transparency_ratio*100:.1f}% transparency - skipping")
        return original_img.convert('RGBA')
    
    # Step 1: Apply rembg
    print("  Step 1: Applying rembg...")
    nobg_img = remove(original_img, alpha_matting=alpha_matting)
    
    # Step 2: Detect what background color was removed
    print("  Step 2: Detecting background color...")
    bg_color, std_dev = detect_removed_background_color(original_img, nobg_img)
    
    if bg_color:
        if std_dev < 10:
            print(f"  Detected uniform background: RGB{bg_color} (std: {std_dev:.1f})")
        else:
            print(f"  Detected varied background: RGB{bg_color} (std: {std_dev:.1f})")
        
        # Step 3: Clean up false negatives
        print("  Step 3: Cleaning up false negatives...")
        cleaned_img, cleaned_count = clean_false_negatives(nobg_img, bg_color, std_dev)
        
        if cleaned_count > 0:
            print(f"  ✓ Cleaned up {cleaned_count} false negative pixels")
        
        return cleaned_img
    else:
        print("  No clear background color detected")
        return nobg_img

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        input_path = sys.argv[1]
        result = smart_remove_background(input_path)
        output_path = input_path.replace('.png', '_nobg.png')
        result.save(output_path)
        print(f"Saved to: {output_path}")
    else:
        print("Usage: python smart_transparency_image.py <image.png>")