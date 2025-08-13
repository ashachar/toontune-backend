#!/usr/bin/env python3
"""
Simple approach: Threshold for black/dark pixels, then skeletonize
Much simpler than using Anime2Sketch!
"""

import cv2
import numpy as np
from skimage.morphology import skeletonize, thin
import sys

def simple_black_extraction(input_path, output_path, black_threshold=50):
    """
    Extract black/dark lines from image and skeletonize them
    
    Args:
        input_path: Input image path
        output_path: Output image path  
        black_threshold: Pixels darker than this are considered black (0-255)
    """
    
    # Read image
    img = cv2.imread(input_path)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Method 1: Simple threshold for black pixels
    # Find pixels darker than threshold
    black_mask = gray < black_threshold
    
    # Convert to binary image (white lines on black background)
    binary = black_mask.astype(np.uint8) * 255
    
    # Optional: Clean up small noise
    kernel = np.ones((2,2), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    # Skeletonize to get 1-pixel lines
    skeleton = skeletonize(binary > 0)
    
    # Convert back to image format
    result = (1 - skeleton) * 255  # Invert so lines are black on white
    result = result.astype(np.uint8)
    
    cv2.imwrite(output_path, result)
    print(f"Saved simple threshold + skeleton to {output_path}")
    
    # Also save intermediate steps
    cv2.imwrite(output_path.replace('.png', '_mask.png'), binary)
    
    return result

def adaptive_black_extraction(input_path, output_path):
    """
    Use adaptive thresholding to handle varying lighting
    """
    
    # Read image
    img = cv2.imread(input_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Method 2: Adaptive threshold to find dark regions
    # This adapts to local brightness variations
    binary = cv2.adaptiveThreshold(gray, 255,
                                  cv2.ADAPTIVE_THRESH_MEAN_C,
                                  cv2.THRESH_BINARY,
                                  blockSize=11,
                                  C=15)
    
    # Invert so black pixels become white
    binary = 255 - binary
    
    # Clean up
    kernel = np.ones((2,2), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    # Skeletonize
    skeleton = skeletonize(binary > 0)
    
    # Convert back
    result = (1 - skeleton) * 255
    result = result.astype(np.uint8)
    
    cv2.imwrite(output_path, result)
    print(f"Saved adaptive threshold + skeleton to {output_path}")
    
    return result

def color_based_extraction(input_path, output_path, color_threshold=60):
    """
    Extract based on color darkness (works well for colored images)
    """
    
    # Read image in color
    img = cv2.imread(input_path)
    
    # Calculate pixel darkness (low values in all channels = dark/black)
    # Use maximum across channels - if any channel is bright, pixel isn't black
    max_channel = np.max(img, axis=2)
    
    # Find truly dark pixels (dark in ALL channels)
    dark_mask = max_channel < color_threshold
    
    # Convert to binary
    binary = dark_mask.astype(np.uint8) * 255
    
    # Clean up
    kernel = np.ones((2,2), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    # Skeletonize
    skeleton = skeletonize(binary > 0)
    
    # Convert back
    result = (1 - skeleton) * 255
    result = result.astype(np.uint8)
    
    cv2.imwrite(output_path, result)
    print(f"Saved color-based extraction + skeleton to {output_path}")
    
    # Save mask for debugging
    cv2.imwrite(output_path.replace('.png', '_color_mask.png'), binary)
    
    return result

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python simple_threshold_skeleton.py input.png output.png [method] [threshold]")
        print("Methods: simple (default), adaptive, color")
        print("Threshold: 0-255 for black threshold (default 50)")
        sys.exit(1)
    
    method = sys.argv[3] if len(sys.argv) > 3 else "simple"
    threshold = int(sys.argv[4]) if len(sys.argv) > 4 else 50
    
    if method == "adaptive":
        adaptive_black_extraction(sys.argv[1], sys.argv[2])
    elif method == "color":
        color_based_extraction(sys.argv[1], sys.argv[2], threshold)
    else:
        simple_black_extraction(sys.argv[1], sys.argv[2], threshold)