#!/usr/bin/env python3

import cv2
import numpy as np
import subprocess
import os

def analyze_sea_regions(image_path):
    """
    Analyze the sea video frame to identify water regions.
    """
    print(f"📊 Analyzing sea regions in: {image_path}")
    
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Could not load image")
        return None
    
    height, width = img.shape[:2]
    print(f"Image dimensions: {width}x{height}")
    
    # Convert to HSV for better color detection
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Detect water (blueish/gray colors in the sea)
    # The sea appears to be grayish-blue
    # Lower mask for darker water
    lower_water1 = np.array([0, 0, 50])  # Gray tones
    upper_water1 = np.array([180, 30, 150])
    
    # Upper mask for lighter water/horizon
    lower_water2 = np.array([90, 20, 100])  # Light blue
    upper_water2 = np.array([130, 60, 200])
    
    # Create masks
    mask1 = cv2.inRange(hsv, lower_water1, upper_water1)
    mask2 = cv2.inRange(hsv, lower_water2, upper_water2)
    water_mask = cv2.bitwise_or(mask1, mask2)
    
    # Clean up the mask
    kernel = np.ones((5,5), np.uint8)
    water_mask = cv2.morphologyEx(water_mask, cv2.MORPH_CLOSE, kernel)
    water_mask = cv2.morphologyEx(water_mask, cv2.MORPH_OPEN, kernel)
    
    # Find the water region boundaries
    print("\n🌊 Analyzing water boundaries:")
    
    # Scan from middle of image horizontally
    mid_y = height // 2
    water_pixels = []
    
    for y in range(height):
        row = water_mask[y, :]
        water_in_row = np.where(row > 0)[0]
        if len(water_in_row) > 100:  # Significant water in this row
            water_pixels.append((y, len(water_in_row)))
    
    if water_pixels:
        # Find the main water body
        water_pixels.sort(key=lambda x: x[1], reverse=True)
        
        # Get continuous water region
        water_top = height
        water_bottom = 0
        
        for y, count in water_pixels:
            if count > width * 0.3:  # At least 30% of width is water
                water_top = min(water_top, y)
                water_bottom = max(water_bottom, y)
        
        print(f"Water region found:")
        print(f"  Top boundary: y={water_top}")
        print(f"  Bottom boundary: y={water_bottom}")
        print(f"  Water height: {water_bottom - water_top}px")
        
        # The horizontal center of the water
        mid_x = width // 2
        
        # Check water at different x positions
        for x in [width//4, width//2, 3*width//4]:
            col = water_mask[:, x]
            water_in_col = np.where(col > 0)[0]
            if len(water_in_col) > 0:
                print(f"  At x={x}: water from y={water_in_col[0]} to y={water_in_col[-1]}")
        
        return {
            'water_top': water_top,
            'water_bottom': water_bottom,
            'water_center_x': mid_x,
            'mask': water_mask
        }
    
    return None

def create_visual_analysis(image_path, output_dir):
    """
    Create visual analysis showing water regions.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    img = cv2.imread(image_path)
    height, width = img.shape[:2]
    
    # Analyze the image visually
    print("\n📍 Visual Analysis of Sea Frame:")
    print(f"  Total height: {height}px")
    print(f"  Total width: {width}px")
    
    # Looking at the image:
    # - Sky/horizon is at the top (light green/blue area)
    # - The actual water/sea is the gray area in the middle
    # - Beach/sand is at the bottom
    
    # Based on visual inspection of the frame:
    # The water appears to be approximately:
    # - Top edge (horizon): around y=140-150
    # - Bottom edge (beach): around y=280-290
    # - So water occupies roughly y=150 to y=280
    
    water_top = 150  # Where water meets horizon
    water_bottom = 280  # Where water meets beach
    water_middle = (water_top + water_bottom) // 2  # Middle of water
    
    print(f"\n🎯 Identified water region (by visual inspection):")
    print(f"  Water top (horizon): y={water_top}")
    print(f"  Water middle: y={water_middle}")
    print(f"  Water bottom (beach): y={water_bottom}")
    print(f"  Water depth: {water_bottom - water_top}px")
    
    # Create annotated image
    annotated = img.copy()
    
    # Draw water boundaries
    cv2.line(annotated, (0, water_top), (width, water_top), (0, 255, 0), 2)
    cv2.putText(annotated, "Water Top/Horizon", (10, water_top-5), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    cv2.line(annotated, (0, water_middle), (width, water_middle), (255, 255, 0), 2)
    cv2.putText(annotated, "Water Middle", (10, water_middle-5), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    
    cv2.line(annotated, (0, water_bottom), (width, water_bottom), (0, 0, 255), 2)
    cv2.putText(annotated, "Water Bottom/Beach", (10, water_bottom-5), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
    # Draw center line
    cv2.line(annotated, (width//2, 0), (width//2, height), (255, 0, 255), 1)
    cv2.putText(annotated, "Center", (width//2+5, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
    
    # Mark where sailor should emerge
    emerge_x = width // 2
    emerge_y = water_middle
    cv2.circle(annotated, (emerge_x, emerge_y), 10, (0, 255, 255), -1)
    cv2.putText(annotated, "Sailor emerges here", (emerge_x+15, emerge_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    
    # Save annotated image
    output_path = os.path.join(output_dir, "sea_analysis.png")
    cv2.imwrite(output_path, annotated)
    print(f"\n📸 Saved annotated analysis: {output_path}")
    
    # Create a simple water mask
    water_mask = np.zeros((height, width), dtype=np.uint8)
    water_mask[water_top:water_bottom, :] = 255
    
    mask_path = os.path.join(output_dir, "water_mask.png")
    cv2.imwrite(mask_path, water_mask)
    print(f"📸 Saved water mask: {mask_path}")
    
    return {
        'water_top': water_top,
        'water_middle': water_middle,
        'water_bottom': water_bottom,
        'center_x': width // 2,
        'width': width,
        'height': height
    }

def main():
    image_path = "output/sea_first_frame.png"
    output_dir = "output/sea_analysis"
    
    if not os.path.exists(image_path):
        print(f"Error: {image_path} not found")
        return
    
    print("=" * 60)
    print("SEA WATER REGION ANALYSIS")
    print("=" * 60)
    
    # Analyze the sea
    water_info = create_visual_analysis(image_path, output_dir)
    
    if water_info:
        print("\n" + "=" * 60)
        print("💡 EXPLANATION OF WATER DETECTION:")
        print("=" * 60)
        print("\nHow I figured out the water region:")
        print("1. Visually inspected the frame")
        print("2. Identified three distinct regions:")
        print("   - Top: Sky/horizon (light green area)")
        print("   - Middle: Water/sea (gray area)")
        print("   - Bottom: Beach/sand (beige area)")
        print("3. The water occupies the middle third of the image")
        print("4. Water boundaries:")
        print(f"   - Top edge at y={water_info['water_top']} (horizon line)")
        print(f"   - Bottom edge at y={water_info['water_bottom']} (beach line)")
        print(f"5. For realistic emergence:")
        print(f"   - Sailor should start at y={water_info['water_middle']} (middle of water)")
        print(f"   - Position at x={water_info['center_x']} (center of screen)")
        print(f"   - Rise up from this point with only head visible initially")
        
        print("\n✅ Ready to create proper water emergence effect!")
        print(f"   Check {output_dir}/ for visual analysis")

if __name__ == "__main__":
    main()