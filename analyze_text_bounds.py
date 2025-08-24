#!/usr/bin/env python3
"""Analyze exact text bounds - detect only bright yellow front face"""

import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

print("="*80)
print("ANALYZING TEXT BOUNDS - BRIGHT YELLOW ONLY")
print("="*80)

# Load the generated frames
frames_to_check = [
    ('verify_last_motion.png', 'Last Motion'),
    ('verify_first_dissolve.png', 'First Dissolve'),
]

def detect_bright_yellow_text(image_path):
    """Detect only the bright yellow front face of text"""
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not load {image_path}")
        return None
    
    # Convert to RGB and HSV
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Detect bright yellow (tighter range for front face only)
    # The front face is brighter than the depth layers
    lower_yellow = np.array([25, 150, 200])  # Higher saturation and value
    upper_yellow = np.array([35, 255, 255])
    
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
    # Clean up mask
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Get bounding box of all contours
        all_points = np.concatenate(contours)
        x, y, w, h = cv2.boundingRect(all_points)
        
        # Also find individual letter bounds
        letter_bounds = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 100:  # Filter small noise
                lx, ly, lw, lh = cv2.boundingRect(contour)
                letter_bounds.append((lx, ly, lw, lh))
        
        letter_bounds.sort(key=lambda b: b[0])  # Sort by x position
        
        return {
            'overall': (x, y, w, h),
            'letters': letter_bounds,
            'mask': mask,
            'image': rgb
        }
    
    return None

# Analyze each frame
results = {}
for image_path, label in frames_to_check:
    print(f"\nAnalyzing: {label} ({image_path})")
    result = detect_bright_yellow_text(image_path)
    
    if result:
        x, y, w, h = result['overall']
        center_x = x + w/2
        center_y = y + h/2
        
        print(f"  Overall text bounds:")
        print(f"    Position: ({x}, {y})")
        print(f"    Size: {w}x{h}")
        print(f"    Center: ({center_x:.1f}, {center_y:.1f})")
        print(f"    Left edge: {x}")
        print(f"    Right edge: {x+w}")
        
        if result['letters']:
            print(f"\n  Individual letters detected: {len(result['letters'])}")
            print(f"  First 3 letter positions:")
            for i, (lx, ly, lw, lh) in enumerate(result['letters'][:3]):
                print(f"    Letter {i}: x={lx}, width={lw}")
        
        results[label] = result

# Compare positions
if len(results) == 2:
    print("\n" + "="*80)
    print("POSITION COMPARISON")
    print("="*80)
    
    motion_result = results.get('Last Motion')
    dissolve_result = results.get('First Dissolve')
    
    if motion_result and dissolve_result:
        mx, my, mw, mh = motion_result['overall']
        dx, dy, dw, dh = dissolve_result['overall']
        
        print(f"\nLast Motion:")
        print(f"  Left edge: {mx}")
        print(f"  Center: {mx + mw/2:.1f}")
        print(f"  Right edge: {mx + mw}")
        
        print(f"\nFirst Dissolve:")
        print(f"  Left edge: {dx}")
        print(f"  Center: {dx + dw/2:.1f}")
        print(f"  Right edge: {dx + dw}")
        
        print(f"\nPosition change:")
        print(f"  Left edge shift: {dx - mx} pixels")
        print(f"  Center shift: {(dx + dw/2) - (mx + mw/2):.1f} pixels")
        print(f"  Right edge shift: {(dx + dw) - (mx + mw)} pixels")
        
        if abs(dx - mx) > 5:
            print(f"\n❌ SIGNIFICANT LEFT SHIFT DETECTED: {dx - mx} pixels!")

# Create visualization
print("\n" + "="*80)
print("CREATING VISUALIZATION")
print("="*80)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

for idx, (label, result) in enumerate(results.items()):
    if result:
        # Original image
        axes[idx, 0].imshow(result['image'])
        axes[idx, 0].set_title(f'{label} - Original')
        axes[idx, 0].axis('off')
        
        # Mask
        axes[idx, 1].imshow(result['mask'], cmap='gray')
        axes[idx, 1].set_title(f'{label} - Bright Yellow Mask')
        axes[idx, 1].axis('off')
        
        # Annotated
        annotated = result['image'].copy()
        x, y, w, h = result['overall']
        
        # Draw overall bounds in red
        cv2.rectangle(annotated, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        # Draw center line in green
        cx = int(x + w/2)
        cv2.line(annotated, (cx, 0), (cx, annotated.shape[0]), (0, 255, 0), 1)
        
        # Draw individual letter bounds in blue
        for lx, ly, lw, lh in result['letters'][:5]:  # First 5 letters
            cv2.rectangle(annotated, (lx, ly), (lx+lw, ly+lh), (0, 0, 255), 1)
        
        axes[idx, 2].imshow(annotated)
        axes[idx, 2].set_title(f'{label} - Detected Bounds')
        axes[idx, 2].axis('off')
        
        # Add text annotations
        axes[idx, 2].text(x, y-10, f'Left: {x}', color='red', fontsize=8)
        axes[idx, 2].text(cx, y-10, f'Center: {cx}', color='green', fontsize=8)

plt.suptitle('Text Bounds Analysis - Bright Yellow Detection', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('text_bounds_analysis.png', dpi=150)
print("\nSaved: text_bounds_analysis.png")

print("\n✅ ANALYSIS COMPLETE")