#!/usr/bin/env python3
"""
Test head extraction on the original image (not skeleton)
"""

import cv2
import numpy as np
import sys
import os

def extract_head_from_original(img_path, head_box_ratio=(0.2, 0, 0.8, 0.4)):
    """
    Extract head region from original image by checking connectivity
    """
    print("Extracting head from original image...")
    
    # Read original image
    img = cv2.imread(img_path)
    if img is None:
        print(f"Failed to read {img_path}")
        return None
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    H, W = gray.shape
    print(f"Original image shape: {H}x{W}")
    
    # Define head bounding box
    x0 = int(W * head_box_ratio[0])
    y0 = int(H * head_box_ratio[1])
    x1 = int(W * head_box_ratio[2])
    y1 = int(H * head_box_ratio[3])
    
    print(f"Head box: x={x0}-{x1}, y={y0}-{y1}")
    
    # Extract head region
    head_region = gray[y0:y1, x0:x1].copy()
    
    # Create binary mask (non-white pixels)
    # For cartoon images, background is usually white (255)
    _, binary = cv2.threshold(head_region, 250, 255, cv2.THRESH_BINARY_INV)
    
    # Find connected components
    num_labels, labels = cv2.connectedComponents(binary, connectivity=8)
    print(f"Found {num_labels - 1} components in head region")
    
    if num_labels <= 1:
        print("No components found")
        return None
    
    # Find component sizes
    component_sizes = []
    for label in range(1, num_labels):
        size = np.sum(labels == label)
        component_sizes.append((label, size))
    
    component_sizes.sort(key=lambda x: x[1], reverse=True)
    
    # The largest component should be the main head
    if len(component_sizes) > 0:
        main_label = component_sizes[0][0]
        main_size = component_sizes[0][1]
        print(f"Main head component: label={main_label}, size={main_size} pixels")
        
        # Create mask with the main component (which should include everything connected)
        head_mask_region = (labels == main_label).astype(np.uint8) * 255
        
        # Report on other components
        for label, size in component_sizes[1:5]:  # Show top 5
            print(f"  Other component: label={label}, size={size} pixels")
    else:
        print("No valid components found")
        return None
    
    # Create full-size mask
    full_mask = np.zeros(gray.shape, dtype=np.uint8)
    full_mask[y0:y1, x0:x1] = head_mask_region
    
    # Create visualization
    viz = img.copy()
    
    # Draw head box in blue
    cv2.rectangle(viz, (x0, y0), (x1, y1), (255, 0, 0), 3)
    
    # Highlight head pixels in red
    head_pixels = np.where(full_mask > 0)
    overlay = viz.copy()
    overlay[head_pixels] = [0, 0, 255]
    viz = cv2.addWeighted(viz, 0.3, overlay, 0.7, 0)
    
    return full_mask, viz

def main():
    # Test with the original robot image
    original_path = "../../uploads/assets/robot.png"
    output_dir = "head_original_test"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract head from original
    mask, viz = extract_head_from_original(original_path)
    
    if mask is None:
        print("Failed to extract head")
        return
    
    # Save outputs
    mask_path = os.path.join(output_dir, "robot_head_mask_from_original.png")
    viz_path = os.path.join(output_dir, "robot_head_viz_from_original.png")
    
    cv2.imwrite(mask_path, mask)
    cv2.imwrite(viz_path, viz)
    
    # Count pixels
    head_pixels = np.sum(mask > 0)
    print(f"\nTotal head pixels from original: {head_pixels}")
    
    print(f"\nSaved:")
    print(f"  Mask: {mask_path}")
    print(f"  Visualization: {viz_path}")
    
    # Open visualization
    os.system(f'open {viz_path}')

if __name__ == "__main__":
    main()