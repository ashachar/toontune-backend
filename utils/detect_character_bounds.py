#!/usr/bin/env python3
"""
Detect the bounding box of non-green content in a frame with green screen.
This helps us find where the character actually appears.
"""

import cv2
import numpy as np
import sys


def detect_character_bounds(image_path, green_threshold=0.08):
    """
    Detect the bounding box of non-green content (the character).
    Returns: (x_center, y_center, width, height) of the character
    """
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image {image_path}")
        return None
    
    h, w = img.shape[:2]
    
    # Convert to HSV for better green detection
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Define green color range (typical green screen)
    # Hue around 60 degrees (green), high saturation
    lower_green = np.array([40, 40, 40])
    upper_green = np.array([80, 255, 255])
    
    # Create mask for green pixels
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    
    # Invert to get non-green (character) pixels
    character_mask = cv2.bitwise_not(green_mask)
    
    # Find contours of non-green areas
    contours, _ = cv2.findContours(character_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        print("No character detected (all green)")
        return None
    
    # Find the largest contour (main character)
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Get bounding box
    x, y, char_w, char_h = cv2.boundingRect(largest_contour)
    
    # Calculate center
    x_center = x + char_w // 2
    y_center = y + char_h // 2
    
    print(f"Image dimensions: {w}x{h}")
    print(f"Character detected at:")
    print(f"  Bounding box: x={x}, y={y}, width={char_w}, height={char_h}")
    print(f"  Center: ({x_center}, {y_center})")
    print(f"  Offset from frame center: ({x_center - w//2:+d}, {y_center - h//2:+d})")
    
    # Save visualization
    vis = img.copy()
    cv2.rectangle(vis, (x, y), (x + char_w, y + char_h), (0, 0, 255), 3)
    cv2.circle(vis, (x_center, y_center), 10, (255, 0, 0), -1)
    cv2.circle(vis, (w//2, h//2), 10, (0, 255, 0), -1)  # Frame center in green
    
    output_path = image_path.replace('.png', '_bounds.png')
    cv2.imwrite(output_path, vis)
    print(f"Saved visualization to: {output_path}")
    
    return x_center, y_center, char_w, char_h


if __name__ == "__main__":
    # Test on a frame with the character
    test_image = "outputs/check_character_position.png"
    if len(sys.argv) > 1:
        test_image = sys.argv[1]
    
    result = detect_character_bounds(test_image)
    if result:
        x_center, y_center, width, height = result
        print(f"\nRecommended adjustments for eraser animation:")
        print(f"  center_x = {x_center}  # (was 640)")
        print(f"  center_y = {y_center}  # (was 360)")