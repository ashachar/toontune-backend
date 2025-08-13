#!/usr/bin/env python3
"""
Test head extraction that includes all components INSIDE the main head outline
"""

import cv2
import numpy as np
from skimage.morphology import skeletonize
import sys
import os

def extract_head_with_interior(skeleton_img, head_box_ratio=(0.2, 0, 0.8, 0.4)):
    """
    Extract head region including all components inside the main head outline
    """
    print("Extracting head region with interior features...")
    
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
    
    # Find the largest connected component (main head outline)
    component_sizes = []
    for label in range(1, num_labels):
        size = np.sum(labels == label)
        component_sizes.append((label, size))
    
    component_sizes.sort(key=lambda x: x[1], reverse=True)
    
    # The largest component should be the head outline
    main_label = component_sizes[0][0]
    main_size = component_sizes[0][1]
    
    print(f"Main head outline: label={main_label}, size={main_size} pixels")
    
    # Get the main outline mask
    outline_mask = (labels == main_label).astype(np.uint8)
    
    # Find the convex hull of the main outline to define the "inside" region
    outline_points = np.argwhere(outline_mask)
    if len(outline_points) > 3:
        # Convert to cv2 format (x, y)
        outline_points_cv = outline_points[:, [1, 0]]
        hull = cv2.convexHull(outline_points_cv)
        
        # Create a filled polygon mask from the hull
        hull_mask = np.zeros_like(outline_mask)
        cv2.fillPoly(hull_mask, [hull], 255)
        
        print(f"Created convex hull with {len(hull)} points")
    else:
        # Fallback: use the outline itself
        hull_mask = outline_mask * 255
    
    # Now include all components that are mostly inside this hull
    final_mask = np.zeros_like(head_region)
    
    for label, size in component_sizes:
        component_mask = (labels == label)
        
        # Check how much of this component is inside the hull
        overlap = np.sum(component_mask & (hull_mask > 0))
        overlap_ratio = overlap / size if size > 0 else 0
        
        # Include if more than 50% is inside the hull, or if it's the main outline
        if overlap_ratio > 0.5 or label == main_label:
            final_mask = final_mask | (component_mask * 255)
            
            if label == main_label:
                print(f"  Included: label={label} (HEAD OUTLINE), size={size}")
            else:
                print(f"  Included: label={label} (INSIDE HEAD), size={size}, overlap={overlap_ratio:.1%}")
        else:
            print(f"  Excluded: label={label} (OUTSIDE), size={size}, overlap={overlap_ratio:.1%}")
    
    # Create full-size mask
    full_mask = np.zeros_like(lines)
    full_mask[y0:y1, x0:x1] = final_mask
    
    # Create visualization
    viz = cv2.cvtColor(skeleton_img, cv2.COLOR_GRAY2BGR)
    
    # Draw head box in blue
    cv2.rectangle(viz, (x0, y0), (x1, y1), (255, 0, 0), 2)
    
    # Show different components in different colors
    head_region_colored = cv2.cvtColor(head_region, cv2.COLOR_GRAY2BGR)
    
    for label, size in component_sizes[:10]:  # Show first 10 components
        component_mask = (labels == label)
        
        # Check if included
        overlap = np.sum(component_mask & (hull_mask > 0))
        overlap_ratio = overlap / size if size > 0 else 0
        
        if overlap_ratio > 0.5 or label == main_label:
            # Color based on type
            if label == main_label:
                color = [0, 0, 255]  # Red for outline
            else:
                color = [0, 255, 0]  # Green for interior
            
            # Apply color to visualization
            component_pixels = np.where(component_mask)
            for y, x in zip(component_pixels[0], component_pixels[1]):
                viz[y0 + y, x0 + x] = color
    
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
    
    # Extract head with interior features
    head_mask, viz, head_box = extract_head_with_interior(skeleton_img)
    
    if head_mask is None:
        print("Failed to extract head region")
        return
    
    # Count pixels in head mask
    head_pixel_count = np.sum(head_mask > 0)
    print(f"\nTotal head pixels (outline + interior): {head_pixel_count}")
    
    # Save outputs
    mask_path = os.path.join(output_dir, "robot_head_mask_complete.png")
    viz_path = os.path.join(output_dir, "robot_head_viz_complete.png")
    
    cv2.imwrite(mask_path, head_mask)
    cv2.imwrite(viz_path, viz)
    
    print(f"\nSaved outputs:")
    print(f"  Head mask: {mask_path}")
    print(f"  Visualization: {viz_path}")
    
    # Create side-by-side comparison
    comparison = np.hstack([
        cv2.cvtColor(skeleton_img, cv2.COLOR_GRAY2BGR),
        viz
    ])
    
    # Add labels
    cv2.putText(comparison, "Original Skeleton", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.putText(comparison, "Head Extraction (Red=Outline, Green=Interior)", (1034, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    
    comparison_path = os.path.join(output_dir, "robot_head_comparison_complete.png")
    cv2.imwrite(comparison_path, comparison)
    print(f"  Comparison: {comparison_path}")
    
    return mask_path

if __name__ == "__main__":
    skeleton_path = "robot_euler_output/robot_skeleton.png"
    output_dir = "head_extraction_test_v2"
    
    mask_path = visualize_head_extraction(skeleton_path, output_dir)
    
    if mask_path:
        print(f"\nOpening head mask visualization...")
        os.system(f'open {output_dir}/robot_head_comparison_complete.png')