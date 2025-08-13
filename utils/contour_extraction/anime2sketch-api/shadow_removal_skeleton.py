#!/usr/bin/env python3
"""
Advanced pipeline: Threshold → Remove shadows/gradients → Skeletonize
Gets clean lines without shadow artifacts!
"""

import cv2
import numpy as np
from skimage.morphology import skeletonize, remove_small_objects, remove_small_holes
from scipy import ndimage
import sys

def remove_shadows_and_skeletonize(input_path, output_path, black_threshold=80):
    """
    Extract black lines, remove shadows/gradients, then skeletonize
    """
    
    # Read image
    img = cv2.imread(input_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Step 1: Initial mask of dark pixels
    dark_mask = gray < black_threshold
    
    # Step 2: Identify actual lines vs shadows
    # Lines are typically thin and continuous
    # Shadows are typically thick blobs
    
    # Convert to binary image
    binary = dark_mask.astype(np.uint8) * 255
    
    # Step 3: Edge detection on the mask to find boundaries
    # This helps separate lines from filled areas
    edges = cv2.Canny(binary, 50, 150)
    
    # Step 4: Use morphological operations to identify line-like structures
    # Create different kernels for different orientations
    kernels = [
        cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1)),  # Horizontal
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5)),  # Vertical
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))  # Circular
    ]
    
    # Find line-like structures
    line_mask = np.zeros_like(binary)
    for kernel in kernels:
        # Opening removes small blobs but preserves lines
        opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        # Closing connects broken lines
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
        line_mask = cv2.bitwise_or(line_mask, closed)
    
    # Step 5: Remove large filled areas (likely shadows)
    # Find contours
    contours, _ = cv2.findContours(line_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create clean mask with only line-like contours
    clean_mask = np.zeros_like(binary)
    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        # Calculate "thinness" ratio - lines have high perimeter relative to area
        if perimeter > 0:
            thinness = (perimeter * perimeter) / (4 * np.pi * area + 0.001)
            
            # Keep thin structures (high thinness) and small areas
            if thinness > 5 or area < 500:  # Adjust these thresholds as needed
                cv2.drawContours(clean_mask, [contour], -1, 255, -1)
    
    # Step 6: Combine edge information with clean mask
    combined = cv2.bitwise_or(edges, clean_mask)
    
    # Step 7: Final cleanup
    kernel_small = np.ones((2,2), np.uint8)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel_small)
    
    # Step 8: Skeletonize
    skeleton = skeletonize(combined > 0)
    
    # Convert back to image
    result = (1 - skeleton) * 255
    result = result.astype(np.uint8)
    
    cv2.imwrite(output_path, result)
    print(f"Saved shadow-removed skeleton to {output_path}")
    
    # Save intermediate steps for debugging
    cv2.imwrite(output_path.replace('.png', '_initial_mask.png'), binary)
    cv2.imwrite(output_path.replace('.png', '_clean_mask.png'), clean_mask)
    cv2.imwrite(output_path.replace('.png', '_edges.png'), edges)
    
    return result

def gradient_based_cleaning(input_path, output_path, black_threshold=80):
    """
    Use gradient information to distinguish lines from shadows
    Lines have sharp gradients, shadows have smooth gradients
    """
    
    # Read image
    img = cv2.imread(input_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Step 1: Calculate gradients
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    # Step 2: Find high gradient areas (edges)
    gradient_thresh = np.percentile(gradient_magnitude, 90)  # Top 10% gradients
    edge_mask = gradient_magnitude > gradient_thresh
    
    # Step 3: Find dark areas
    dark_mask = gray < black_threshold
    
    # Step 4: Combine - keep dark pixels that are near edges
    # Dilate edge mask to include nearby pixels
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    edge_region = cv2.dilate(edge_mask.astype(np.uint8) * 255, kernel, iterations=2)
    
    # Keep only dark pixels in edge regions
    lines_only = cv2.bitwise_and(dark_mask.astype(np.uint8) * 255, edge_region)
    
    # Step 5: Clean up
    kernel_small = np.ones((2,2), np.uint8)
    lines_only = cv2.morphologyEx(lines_only, cv2.MORPH_CLOSE, kernel_small)
    lines_only = cv2.morphologyEx(lines_only, cv2.MORPH_OPEN, kernel_small)
    
    # Step 6: Skeletonize
    skeleton = skeletonize(lines_only > 0)
    
    # Convert back
    result = (1 - skeleton) * 255
    result = result.astype(np.uint8)
    
    cv2.imwrite(output_path, result)
    print(f"Saved gradient-based skeleton to {output_path}")
    
    # Save intermediate steps
    cv2.imwrite(output_path.replace('.png', '_gradient.png'), 
                (gradient_magnitude / gradient_magnitude.max() * 255).astype(np.uint8))
    cv2.imwrite(output_path.replace('.png', '_lines_only.png'), lines_only)
    
    return result

def connected_component_filtering(input_path, output_path, black_threshold=80):
    """
    Filter connected components by shape characteristics
    """
    
    # Read image
    img = cv2.imread(input_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Get dark mask
    dark_mask = (gray < black_threshold).astype(np.uint8) * 255
    
    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(dark_mask, connectivity=8)
    
    # Create filtered mask
    filtered_mask = np.zeros_like(dark_mask)
    
    for i in range(1, num_labels):  # Skip background (label 0)
        # Get component stats
        x, y, w, h, area = stats[i]
        
        # Calculate shape characteristics
        aspect_ratio = w / (h + 0.001)
        extent = area / (w * h + 0.001)  # How much of bounding box is filled
        
        # Get component mask
        component_mask = (labels == i).astype(np.uint8) * 255
        
        # Calculate perimeter
        contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            perimeter = cv2.arcLength(contours[0], True)
            circularity = 4 * np.pi * area / (perimeter * perimeter + 0.001)
            
            # Keep components that are:
            # - Line-like (low circularity)
            # - Not too filled (low extent suggests line vs blob)
            # - Reasonable size
            if circularity < 0.5 or extent < 0.7 or area < 200:
                filtered_mask = cv2.bitwise_or(filtered_mask, component_mask)
    
    # Clean up
    kernel = np.ones((2,2), np.uint8)
    filtered_mask = cv2.morphologyEx(filtered_mask, cv2.MORPH_CLOSE, kernel)
    
    # Skeletonize
    skeleton = skeletonize(filtered_mask > 0)
    
    # Convert back
    result = (1 - skeleton) * 255
    result = result.astype(np.uint8)
    
    cv2.imwrite(output_path, result)
    print(f"Saved component-filtered skeleton to {output_path}")
    
    # Save intermediate
    cv2.imwrite(output_path.replace('.png', '_filtered.png'), filtered_mask)
    
    return result

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python shadow_removal_skeleton.py input.png output.png [method] [threshold]")
        print("Methods: contour (default), gradient, component")
        print("Threshold: 0-255 for black threshold (default 80)")
        sys.exit(1)
    
    method = sys.argv[3] if len(sys.argv) > 3 else "contour"
    threshold = int(sys.argv[4]) if len(sys.argv) > 4 else 80
    
    if method == "gradient":
        gradient_based_cleaning(sys.argv[1], sys.argv[2], threshold)
    elif method == "component":
        connected_component_filtering(sys.argv[1], sys.argv[2], threshold)
    else:
        remove_shadows_and_skeletonize(sys.argv[1], sys.argv[2], threshold)