#!/usr/bin/env python3
import cv2
import numpy as np
import sys
from scipy import ndimage

def extract_clean_edges(input_path, output_path):
    """Extract clean, thin edges directly from color image"""
    
    # Read image
    img = cv2.imread(input_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Method 1: Canny edge detection with preprocessing
    # Reduce noise while preserving edges
    blurred = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # Multi-scale edge detection
    edges1 = cv2.Canny(blurred, 30, 100)
    edges2 = cv2.Canny(blurred, 50, 150)
    edges3 = cv2.Canny(blurred, 100, 200)
    
    # Combine edges from different scales
    edges = cv2.bitwise_or(edges1, cv2.bitwise_or(edges2, edges3))
    
    # Clean up small noise
    kernel = np.ones((2,2), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel)
    
    # Invert (black lines on white background)
    result = 255 - edges
    
    cv2.imwrite(output_path, result)
    print(f"Saved edges to {output_path}")
    
    return result

def extract_cartoon_contours(input_path, output_path):
    """Extract contours optimized for cartoon/illustration images"""
    
    # Read image
    img = cv2.imread(input_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Find regions with strong color differences (likely outlines in cartoons)
    # Use adaptive threshold to handle varying lighting
    binary = cv2.adaptiveThreshold(gray, 255, 
                                  cv2.ADAPTIVE_THRESH_MEAN_C,
                                  cv2.THRESH_BINARY, 15, 10)
    
    # Detect edges in the binary image
    edges = cv2.Canny(binary, 50, 150)
    
    # Thin the lines using morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    
    # Repeated erosion and dilation to get thin lines
    for _ in range(2):
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    # Final thinning
    edges = cv2.morphologyEx(edges, cv2.MORPH_GRADIENT, kernel)
    
    # Invert
    result = 255 - edges
    
    cv2.imwrite(output_path, result)
    print(f"Saved cartoon contours to {output_path}")
    
    return result

def holistically_nested_edges(input_path, output_path):
    """Use HED-like approach for edge detection"""
    
    img = cv2.imread(input_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Compute gradients
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    
    # Compute gradient magnitude
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    # Normalize and threshold
    magnitude = np.uint8(255 * magnitude / np.max(magnitude))
    _, edges = cv2.threshold(magnitude, 30, 255, cv2.THRESH_BINARY)
    
    # Thin the edges
    kernel = np.ones((2,2), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_GRADIENT, kernel)
    
    # Invert
    result = 255 - edges
    
    cv2.imwrite(output_path, result)
    print(f"Saved HED-style edges to {output_path}")
    
    return result

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python direct_edge_extraction.py input.png output.png [method]")
        print("Methods: canny (default), cartoon, hed")
        sys.exit(1)
    
    method = sys.argv[3] if len(sys.argv) > 3 else "canny"
    
    if method == "cartoon":
        extract_cartoon_contours(sys.argv[1], sys.argv[2])
    elif method == "hed":
        holistically_nested_edges(sys.argv[1], sys.argv[2])
    else:
        extract_clean_edges(sys.argv[1], sys.argv[2])