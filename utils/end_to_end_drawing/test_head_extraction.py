#!/usr/bin/env python3
"""
Test head extraction with connectivity filtering to exclude disconnected pixels
"""

import cv2
import numpy as np
from skimage.morphology import skeletonize
import sys
import os

def extract_connected_head_region(skeleton_img, head_box_ratio=(0.2, 0, 0.8, 0.4)):
    """
    Extract head region but only include pixels that are connected to the main head blob
    """
    print("Extracting connected head region...")
    
    # Get image dimensions
    H, W = skeleton_img.shape[:2]
    
    # Define head bounding box
    x0 = int(W * head_box_ratio[0])
    y0 = int(H * head_box_ratio[1])
    x1 = int(W * head_box_ratio[2])
    y1 = int(H * head_box_ratio[3])
    
    print(f"Head box: x={x0}-{x1}, y={y0}-{y1}")
    
    # Get skeleton lines (black pixels)
    lines = (skeleton_img < 128).astype(np.uint8) * 255
    
    # Extract head region
    head_region = lines[y0:y1, x0:x1].copy()
    
    # Find connected components in head region
    num_labels, labels = cv2.connectedComponents(head_region, connectivity=8)
    print(f"Found {num_labels - 1} components in head region")
    
    if num_labels <= 1:
        print("No components found in head region")
        return None
    
    # Find the largest connected component (main head blob)
    component_sizes = []
    for label in range(1, num_labels):
        size = np.sum(labels == label)
        component_sizes.append((label, size))
    
    component_sizes.sort(key=lambda x: x[1], reverse=True)
    
    # Create mask with only connected components
    # Start with the largest component
    main_label = component_sizes[0][0]
    main_size = component_sizes[0][1]
    connected_mask = (labels == main_label).astype(np.uint8) * 255
    
    print(f"Main component: label={main_label}, size={main_size} pixels")
    
    # Add other large components that are likely connected
    # (components that are at least 10% the size of the main component)
    threshold_size = main_size * 0.1
    
    for label, size in component_sizes[1:]:
        if size >= threshold_size:
            # Check if this component is near the main component
            component_mask = (labels == label)
            
            # Dilate the main mask slightly to check for proximity
            dilated_main = cv2.dilate(connected_mask, np.ones((3,3), np.uint8), iterations=2)
            
            # Check if there's any overlap after dilation (indicating proximity)
            if np.any(dilated_main & (component_mask * 255)):
                connected_mask = connected_mask | (component_mask * 255)
                print(f"  Added connected component: label={label}, size={size} pixels")
            else:
                print(f"  Excluded disconnected component: label={label}, size={size} pixels")
    
    # Create full-size mask
    full_mask = np.zeros_like(lines)
    full_mask[y0:y1, x0:x1] = connected_mask
    
    # Also create a visualization showing the head box
    viz = cv2.cvtColor(skeleton_img, cv2.COLOR_GRAY2BGR)
    
    # Draw head box in blue
    cv2.rectangle(viz, (x0, y0), (x1, y1), (255, 0, 0), 2)
    
    # Overlay connected head pixels in red
    head_pixels = np.where(full_mask > 0)
    viz[head_pixels] = [0, 0, 255]
    
    return full_mask, viz, (x0, y0, x1, y1)

def visualize_head_extraction(skeleton_path, output_dir):
    """
    Test head extraction and save visualizations
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Read skeleton image
    skeleton_img = cv2.imread(skeleton_path, cv2.IMREAD_GRAYSCALE)
    
    if skeleton_img is None:
        print(f"Failed to read {skeleton_path}")
        return
    
    print(f"Skeleton image shape: {skeleton_img.shape}")
    
    # Extract connected head region
    head_mask, viz, head_box = extract_connected_head_region(skeleton_img)
    
    if head_mask is None:
        print("Failed to extract head region")
        return
    
    # Count pixels in head mask
    head_pixel_count = np.sum(head_mask > 0)
    print(f"Total head pixels (connected only): {head_pixel_count}")
    
    # Save outputs
    mask_path = os.path.join(output_dir, "robot_head_mask.png")
    viz_path = os.path.join(output_dir, "robot_head_viz.png")
    
    cv2.imwrite(mask_path, head_mask)
    cv2.imwrite(viz_path, viz)
    
    print(f"\nSaved outputs:")
    print(f"  Head mask: {mask_path}")
    print(f"  Visualization: {viz_path}")
    
    # Also create a side-by-side comparison
    comparison = np.hstack([
        cv2.cvtColor(skeleton_img, cv2.COLOR_GRAY2BGR),
        viz
    ])
    
    comparison_path = os.path.join(output_dir, "robot_head_comparison.png")
    cv2.imwrite(comparison_path, comparison)
    print(f"  Comparison: {comparison_path}")
    
    return mask_path

if __name__ == "__main__":
    skeleton_path = "robot_euler_output/robot_skeleton.png"
    output_dir = "head_extraction_test"
    
    mask_path = visualize_head_extraction(skeleton_path, output_dir)
    
    if mask_path:
        print(f"\nOpening head mask visualization...")
        os.system(f'open {output_dir}/robot_head_comparison.png')