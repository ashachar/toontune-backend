#!/usr/bin/env python3
"""
Optimized thin line extractor for drawing animation
Produces clean, single-pixel width lines suitable for path tracing
"""

import cv2
import numpy as np
import sys
from scipy.ndimage import binary_erosion, binary_dilation
from skimage.morphology import skeletonize, thin

def extract_ultra_thin_lines(input_path, output_path, method='skeleton'):
    """
    Extract ultra-thin (1-pixel) lines from cartoon/illustration images
    Perfect for drawing animation path generation
    """
    
    # Read image
    img = cv2.imread(input_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Step 1: Initial edge detection
    # Use adaptive threshold for cartoon images
    binary = cv2.adaptiveThreshold(gray, 255, 
                                  cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY, 11, 5)
    
    # Step 2: Clean up noise
    kernel_small = np.ones((2,2), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_small)
    
    # Step 3: Extract edges from binary image
    edges = cv2.Canny(binary, 50, 150)
    
    # Step 4: Ensure connectivity
    kernel_connect = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    edges = cv2.dilate(edges, kernel_connect, iterations=1)
    
    # Step 5: Skeletonize to get 1-pixel lines
    if method == 'skeleton':
        # Convert to binary for skimage
        binary_edges = edges > 0
        # Apply skeletonization
        skeleton = skeletonize(binary_edges)
        result = (skeleton * 255).astype(np.uint8)
    elif method == 'thin':
        # Alternative: use thinning
        binary_edges = edges > 0
        thinned = thin(binary_edges)
        result = (thinned * 255).astype(np.uint8)
    else:
        # Fallback: morphological thinning
        result = edges
        for _ in range(3):
            eroded = cv2.erode(result, kernel_small, iterations=1)
            temp = cv2.dilate(eroded, kernel_small, iterations=1)
            result = cv2.subtract(result, temp)
            result = cv2.bitwise_or(result, eroded)
    
    # Step 6: Invert (black lines on white background)
    result = 255 - result
    
    # Step 7: Optional - remove isolated pixels
    kernel_clean = np.array([[0,1,0],[1,1,1],[0,1,0]], dtype=np.uint8)
    opened = cv2.morphologyEx(255-result, cv2.MORPH_OPEN, kernel_clean)
    result = 255 - opened
    
    cv2.imwrite(output_path, result)
    print(f"Saved ultra-thin lines to {output_path}")
    
    # Save intermediate steps for debugging
    cv2.imwrite(output_path.replace('.png', '_binary.png'), binary)
    cv2.imwrite(output_path.replace('.png', '_edges.png'), edges)
    
    return result

def extract_from_sketch(sketch_path, output_path):
    """
    Extract thin lines from Anime2Sketch output
    """
    
    # Read sketch image
    img = cv2.imread(sketch_path, cv2.IMREAD_GRAYSCALE)
    
    # Threshold to get only black lines
    _, binary = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)
    
    # Invert so lines are white
    binary = 255 - binary
    
    # Remove small noise
    kernel = np.ones((2,2), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    # Skeletonize
    skeleton = skeletonize(binary > 0)
    result = (skeleton * 255).astype(np.uint8)
    
    # Invert back
    result = 255 - result
    
    cv2.imwrite(output_path, result)
    print(f"Saved thin lines from sketch to {output_path}")
    
    return result

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python thin_line_extractor.py input.png output.png [method]")
        print("Methods: skeleton (default), thin, morph, sketch")
        print("Use 'sketch' method for processing Anime2Sketch output")
        sys.exit(1)
    
    method = sys.argv[3] if len(sys.argv) > 3 else "skeleton"
    
    if method == "sketch":
        extract_from_sketch(sys.argv[1], sys.argv[2])
    else:
        extract_ultra_thin_lines(sys.argv[1], sys.argv[2], method)