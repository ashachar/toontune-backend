#!/usr/bin/env python3
"""Extract first frame and verify baseline alignment."""

import cv2
import numpy as np
from PIL import Image, ImageDraw

# Extract first frame from the video
cap = cv2.VideoCapture('outputs/test_baseline_bottom_aligned_hq.mp4')
ret, frame = cap.read()
cap.release()

if ret:
    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame_rgb)
    
    # Save the original frame
    img.save('outputs/first_frame_original.png')
    print("✅ Saved original first frame to outputs/first_frame_original.png")
    
    # Create verification image with baseline markers
    verification = img.copy()
    draw = ImageDraw.Draw(verification)
    
    # Detect yellow pixels (the letters)
    frame_array = np.array(img)
    
    # Find yellow pixels (high R and G, low B)
    yellow_mask = (frame_array[:, :, 0] > 200) & (frame_array[:, :, 1] > 200) & (frame_array[:, :, 2] < 100)
    
    # Find the bottom-most yellow pixel for each letter region
    y_coords, x_coords = np.where(yellow_mask)
    
    if len(y_coords) > 0:
        # Find distinct letter regions by clustering x coordinates
        x_sorted = sorted(x_coords)
        letter_regions = []
        current_region = [x_sorted[0]]
        
        for x in x_sorted[1:]:
            if x - current_region[-1] < 10:  # Within same letter
                current_region.append(x)
            else:  # New letter
                letter_regions.append(current_region)
                current_region = [x]
        letter_regions.append(current_region)
        
        # Find bottom of each letter region
        letter_bottoms = []
        for region in letter_regions[:11]:  # Just check "Hello World" (11 letters including space)
            region_x_min = min(region)
            region_x_max = max(region)
            
            # Find all pixels in this x range
            region_pixels = [(x, y) for x, y in zip(x_coords, y_coords) 
                            if region_x_min <= x <= region_x_max]
            
            if region_pixels:
                # Find the bottom-most y coordinate
                bottom_y = max(p[1] for p in region_pixels)
                center_x = (region_x_min + region_x_max) // 2
                letter_bottoms.append((center_x, bottom_y))
        
        # Draw baseline through the average bottom position
        if letter_bottoms:
            avg_bottom = int(np.mean([b[1] for b in letter_bottoms]))
            
            # Draw main baseline
            draw.line([(0, avg_bottom), (img.width, avg_bottom)], fill=(255, 0, 0), width=2)
            
            # Draw markers for each letter's actual bottom
            for center_x, bottom_y in letter_bottoms:
                # Draw a vertical line from the letter bottom to the baseline
                if bottom_y != avg_bottom:
                    draw.line([(center_x, bottom_y), (center_x, avg_bottom)], 
                             fill=(0, 255, 0), width=1)
                
                # Mark the actual bottom point
                draw.ellipse([(center_x-3, bottom_y-3), (center_x+3, bottom_y+3)], 
                           fill=(0, 255, 0), outline=(0, 128, 0))
            
            # Add text annotations
            draw.text((10, 10), f"Red line = Expected baseline (y={avg_bottom})", fill=(255, 0, 0))
            draw.text((10, 30), "Green dots = Actual letter bottoms", fill=(0, 255, 0))
            
            # Calculate and show deviation
            deviations = [abs(bottom_y - avg_bottom) for _, bottom_y in letter_bottoms]
            max_deviation = max(deviations)
            avg_deviation = np.mean(deviations)
            
            draw.text((10, 50), f"Max deviation: {max_deviation} pixels", fill=(0, 0, 0))
            draw.text((10, 70), f"Avg deviation: {avg_deviation:.1f} pixels", fill=(0, 0, 0))
            
            if max_deviation <= 5:
                draw.text((10, 90), "✅ BASELINE ALIGNED!", fill=(0, 128, 0))
            else:
                draw.text((10, 90), "❌ NOT ALIGNED - Letters at different heights!", fill=(255, 0, 0))
            
            # Print analysis
            print(f"\n📊 Baseline Analysis:")
            print(f"Found {len(letter_bottoms)} letters")
            print(f"Average baseline Y: {avg_bottom}")
            print(f"Letter bottom Y positions: {[b[1] for b in letter_bottoms]}")
            print(f"Max deviation from baseline: {max_deviation} pixels")
            print(f"Average deviation: {avg_deviation:.1f} pixels")
            
            if max_deviation <= 5:
                print("✅ Letters are baseline-aligned!")
            else:
                print("❌ Letters are NOT properly aligned!")
    
    verification.save('outputs/first_frame_baseline_verification.png')
    print("\n✅ Saved verification image to outputs/first_frame_baseline_verification.png")
else:
    print("❌ Could not read video frame")