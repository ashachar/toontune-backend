#!/usr/bin/env python3
"""
Detect head using color-based segmentation for cartoon images
"""

import cv2
import numpy as np
import sys
import os

def detect_head_by_color(img_path):
    """
    Detect head region by finding the largest flesh/head-colored region in upper portion
    """
    print("Detecting head using color analysis...")
    
    # Read original image
    img = cv2.imread(img_path)
    if img is None:
        print(f"Failed to read {img_path}")
        return None
    
    H, W = img.shape[:2]
    print(f"Image shape: {H}x{W}")
    
    # For cartoon robot, the head appears to be pinkish/reddish
    # Let's work in HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Define color ranges for potential head colors
    # For this robot, the head appears pinkish
    # We'll look for pinkish/reddish hues
    lower_pink = np.array([160, 50, 50])  # Lower bound for pink/red in HSV
    upper_pink = np.array([180, 255, 255])  # Upper bound
    mask1 = cv2.inRange(hsv, lower_pink, upper_pink)
    
    # Also check for light pink (lower saturation)
    lower_pink2 = np.array([0, 50, 50])
    upper_pink2 = np.array([20, 255, 255])
    mask2 = cv2.inRange(hsv, lower_pink2, upper_pink2)
    
    # Combine masks
    color_mask = cv2.bitwise_or(mask1, mask2)
    
    # Apply morphology to clean up
    kernel = np.ones((5,5), np.uint8)
    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel)
    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel)
    
    # Find connected components in the color mask
    num_labels, labels = cv2.connectedComponents(color_mask, connectivity=8)
    print(f"Found {num_labels - 1} pink/red regions")
    
    # Find the component in the upper portion of image
    best_head = None
    best_score = 0
    
    for label in range(1, num_labels):
        mask = (labels == label)
        area = np.sum(mask)
        
        if area < 1000:  # Skip tiny regions
            continue
        
        # Get bounding box
        points = np.argwhere(mask)
        y_min, x_min = points.min(axis=0)
        y_max, x_max = points.max(axis=0)
        
        center_y = (y_min + y_max) / 2
        width = x_max - x_min
        height = y_max - y_min
        
        # Score based on:
        # - Being in upper half of image
        # - Size
        # - Aspect ratio close to 1 (round/square)
        upper_score = max(0, 1 - (center_y / H))  # Higher score for upper position
        size_score = min(1, area / (H * W * 0.1))  # Normalize by 10% of image
        aspect = width / height if height > 0 else 0
        aspect_score = 1 - abs(1 - aspect) if aspect > 0 else 0
        
        total_score = upper_score * 2 + size_score + aspect_score
        
        print(f"Region {label}: area={area}, pos={center_y/H:.2f}, "
              f"aspect={aspect:.2f}, score={total_score:.2f}")
        
        if total_score > best_score:
            best_score = total_score
            best_head = {
                'label': label,
                'bbox': (x_min, y_min, x_max, y_max),
                'area': area
            }
    
    if best_head is None:
        print("No suitable head region found by color")
        # Fallback: use upper portion heuristic
        upper_region = img[:int(H*0.45), :]
        return detect_head_by_position(img_path)
    
    # Create head mask
    head_mask = (labels == best_head['label']).astype(np.uint8) * 255
    
    # Create visualization
    viz = img.copy()
    x_min, y_min, x_max, y_max = best_head['bbox']
    
    # Add padding
    padding = 20
    x_min = max(0, x_min - padding)
    y_min = max(0, y_min - padding)
    x_max = min(W, x_max + padding)
    y_max = min(H, y_max + padding)
    
    # Draw box
    cv2.rectangle(viz, (x_min, y_min), (x_max, y_max), (0, 255, 0), 3)
    
    # Highlight head
    overlay = viz.copy()
    overlay[head_mask > 0] = [0, 0, 255]
    viz = cv2.addWeighted(viz, 0.3, overlay, 0.7, 0)
    
    return head_mask, viz, (x_min, y_min, x_max, y_max)

def detect_head_by_position(img_path):
    """
    Fallback: detect head by position and geometry
    """
    print("Using position-based head detection...")
    
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape
    
    # Focus on upper 45% of image
    upper_portion = gray[:int(H*0.45), :]
    
    # Find non-background pixels
    _, binary = cv2.threshold(upper_portion, 250, 255, cv2.THRESH_BINARY_INV)
    
    # Get connected components
    num_labels, labels = cv2.connectedComponents(binary, connectivity=8)
    
    # Find largest component in upper region
    largest_area = 0
    largest_label = 0
    
    for label in range(1, num_labels):
        area = np.sum(labels == label)
        if area > largest_area:
            largest_area = area
            largest_label = label
    
    if largest_label == 0:
        return None
    
    # Create mask
    head_mask_upper = (labels == largest_label).astype(np.uint8) * 255
    head_mask = np.zeros(gray.shape, dtype=np.uint8)
    head_mask[:int(H*0.45), :] = head_mask_upper
    
    # Get bounding box
    points = np.argwhere(head_mask > 0)
    if len(points) > 0:
        y_min, x_min = points.min(axis=0)
        y_max, x_max = points.max(axis=0)
        
        # Add padding
        padding = 20
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(W, x_max + padding)
        y_max = min(H, y_max + padding)
        
        # Create viz
        viz = img.copy()
        cv2.rectangle(viz, (x_min, y_min), (x_max, y_max), (0, 255, 0), 3)
        overlay = viz.copy()
        overlay[head_mask > 0] = [0, 0, 255]
        viz = cv2.addWeighted(viz, 0.3, overlay, 0.7, 0)
        
        return head_mask, viz, (x_min, y_min, x_max, y_max)
    
    return None

def main():
    original_path = "../../uploads/assets/robot.png"
    output_dir = "head_color_detect"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Try color-based detection
    result = detect_head_by_color(original_path)
    
    if result is None:
        print("Failed to detect head")
        return
    
    mask, viz, bbox = result
    
    # Save outputs
    mask_path = os.path.join(output_dir, "robot_head_mask_color.png")
    viz_path = os.path.join(output_dir, "robot_head_viz_color.png")
    
    cv2.imwrite(mask_path, mask)
    cv2.imwrite(viz_path, viz)
    
    print(f"\nHead bbox: {bbox}")
    print(f"Saved: {viz_path}")
    
    os.system(f'open {viz_path}')

if __name__ == "__main__":
    main()