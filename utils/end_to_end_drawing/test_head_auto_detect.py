#!/usr/bin/env python3
"""
Automatically detect head region using better heuristics
"""

import cv2
import numpy as np
import sys
import os

def auto_detect_head_region(img_path):
    """
    Automatically detect head region using component analysis
    """
    print("Auto-detecting head region from original image...")
    
    # Read original image
    img = cv2.imread(img_path)
    if img is None:
        print(f"Failed to read {img_path}")
        return None
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    H, W = gray.shape
    print(f"Original image shape: {H}x{W}")
    
    # Create binary mask (non-white pixels)
    _, binary = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)
    
    # Find all connected components in the full image
    num_labels, labels = cv2.connectedComponents(binary, connectivity=8)
    print(f"Found {num_labels - 1} total components in image")
    
    # Analyze each component
    components_info = []
    for label in range(1, num_labels):
        mask = (labels == label).astype(np.uint8)
        
        # Get bounding box
        points = np.argwhere(mask)
        if len(points) == 0:
            continue
            
        y_min, x_min = points.min(axis=0)
        y_max, x_max = points.max(axis=0)
        
        width = x_max - x_min
        height = y_max - y_min
        area = np.sum(mask)
        center_y = (y_min + y_max) / 2
        center_x = (x_min + x_max) / 2
        
        # Calculate aspect ratio
        aspect_ratio = width / height if height > 0 else 0
        
        components_info.append({
            'label': label,
            'area': area,
            'bbox': (x_min, y_min, x_max, y_max),
            'center': (center_x, center_y),
            'width': width,
            'height': height,
            'aspect_ratio': aspect_ratio,
            'top_position': y_min / H  # Normalized position from top
        })
    
    # Sort by area
    components_info.sort(key=lambda x: x['area'], reverse=True)
    
    # Find the head component - typically:
    # 1. In the upper portion of the image (top 50%)
    # 2. Has a roughly circular/square aspect ratio (0.7 - 1.3)
    # 3. Large enough to be significant
    
    head_component = None
    for comp in components_info:
        # Head heuristics
        is_upper = comp['top_position'] < 0.5  # In upper half
        is_round = 0.6 < comp['aspect_ratio'] < 1.5  # Roughly round/square
        is_large = comp['area'] > (H * W * 0.02)  # At least 2% of image
        
        print(f"Component {comp['label']}: area={comp['area']}, "
              f"pos={comp['top_position']:.2f}, aspect={comp['aspect_ratio']:.2f}, "
              f"bbox=({comp['bbox'][0]},{comp['bbox'][1]})-({comp['bbox'][2]},{comp['bbox'][3]})")
        
        if is_upper and is_round and is_large:
            head_component = comp
            print(f"  -> Selected as HEAD")
            break
        else:
            reasons = []
            if not is_upper: reasons.append("too low")
            if not is_round: reasons.append(f"wrong shape")
            if not is_large: reasons.append("too small")
            print(f"  -> Rejected: {', '.join(reasons)}")
    
    if head_component is None:
        print("Could not identify head component automatically")
        # Fallback to largest component in upper half
        for comp in components_info:
            if comp['top_position'] < 0.6:
                head_component = comp
                print(f"Fallback: using component {comp['label']} as head")
                break
    
    if head_component is None:
        return None
    
    # Extract the head mask
    head_mask = (labels == head_component['label']).astype(np.uint8) * 255
    
    # Get tight bounding box with some padding
    x_min, y_min, x_max, y_max = head_component['bbox']
    padding = 10
    x_min = max(0, x_min - padding)
    y_min = max(0, y_min - padding)
    x_max = min(W, x_max + padding)
    y_max = min(H, y_max + padding)
    
    # Create visualization
    viz = img.copy()
    
    # Draw detected head box in green
    cv2.rectangle(viz, (x_min, y_min), (x_max, y_max), (0, 255, 0), 3)
    
    # Highlight head pixels in red
    head_pixels = np.where(head_mask > 0)
    overlay = viz.copy()
    overlay[head_pixels] = [0, 0, 255]
    viz = cv2.addWeighted(viz, 0.3, overlay, 0.7, 0)
    
    # Add text
    cv2.putText(viz, f"Head: Component {head_component['label']}", (x_min, y_min - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(viz, f"Area: {head_component['area']} pixels", (x_min, y_min - 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    return head_mask, viz, (x_min, y_min, x_max, y_max)

def main():
    # Test with the original robot image
    original_path = "../../uploads/assets/robot.png"
    output_dir = "head_auto_detect"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Auto-detect head
    result = auto_detect_head_region(original_path)
    
    if result is None:
        print("Failed to detect head")
        return
    
    mask, viz, bbox = result
    
    # Save outputs
    mask_path = os.path.join(output_dir, "robot_head_mask_auto.png")
    viz_path = os.path.join(output_dir, "robot_head_viz_auto.png")
    
    cv2.imwrite(mask_path, mask)
    cv2.imwrite(viz_path, viz)
    
    # Count pixels
    head_pixels = np.sum(mask > 0)
    print(f"\nTotal head pixels: {head_pixels}")
    print(f"Head bounding box: {bbox}")
    
    print(f"\nSaved:")
    print(f"  Mask: {mask_path}")
    print(f"  Visualization: {viz_path}")
    
    # Open visualization
    os.system(f'open {viz_path}')

if __name__ == "__main__":
    main()