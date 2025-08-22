#!/usr/bin/env python3
"""Detailed test of W sprite extraction and connected components."""

import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageChops, ImageFont
from scipy import ndimage
import os

# Simulate the exact sprite extraction for W
def extract_w_sprite():
    # Load a font (use the same as word_dissolve)
    font_size = 150
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", font_size)
    except:
        font = ImageFont.load_default()
    
    # Create full "HELLO WORLD" text to match video dimensions
    W, H = 1280, 492  # Match video dimensions
    text_x, text_y = 50, 180  # Approximate position
    
    # Extract W using prefix difference method (same as word_dissolve)
    prefix = "HELLO "  # Everything before W
    
    # Render prefix
    imgA = Image.new('RGBA', (W, H), (0, 0, 0, 0))
    dA = ImageDraw.Draw(imgA)
    dA.text((text_x, text_y), prefix, font=font, fill=(255, 255, 0, 255))
    
    # Render prefix + W
    imgB = Image.new('RGBA', (W, H), (0, 0, 0, 0))
    dB = ImageDraw.Draw(imgB)
    dB.text((text_x, text_y), prefix + "W", font=font, fill=(255, 255, 0, 255))
    
    # Get the difference (this is the W sprite)
    arrA = np.array(imgA)
    arrB = np.array(imgB)
    
    # Create mask where B has pixels but A doesn't
    mask = (arrB[:, :, 3] > 0) & (arrA[:, :, 3] == 0)
    
    # Extract W sprite
    w_sprite = np.zeros_like(arrB)
    w_sprite[mask] = arrB[mask]
    
    # Find bounding box
    alpha = w_sprite[:, :, 3]
    ys, xs = np.where(alpha > 0)
    if len(xs) > 0:
        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()
        
        # Crop to bounding box with some padding
        pad = 10
        x1 = max(0, x_min - pad)
        x2 = min(W, x_max + pad + 1)
        y1 = max(0, y_min - pad)
        y2 = min(H, y_max + pad + 1)
        
        cropped_sprite = w_sprite[y1:y2, x1:x2]
        
        return cropped_sprite, (x1, y1, x2, y2)
    
    return w_sprite, (0, 0, W, H)

# Extract W sprite
print("Extracting W sprite using prefix difference method...")
w_sprite, bbox = extract_w_sprite()
print(f"W sprite size: {w_sprite.shape[:2]}, bbox: {bbox}")

# Analyze connected components
alpha = w_sprite[:, :, 3]
alpha_threshold = 5

# Original alpha (before component removal)
alpha_binary_orig = alpha > alpha_threshold

# Find connected components
labeled, num_features = ndimage.label(alpha_binary_orig)
print(f"\nFound {num_features} connected components in W")

# Analyze each component
components = []
for i in range(1, num_features + 1):
    comp_mask = labeled == i
    size = np.sum(comp_mask)
    
    # Find position
    ys, xs = np.where(comp_mask)
    if len(xs) > 0:
        center_x = np.mean(xs)
        center_y = np.mean(ys)
        components.append({
            'id': i,
            'size': size,
            'center': (center_x, center_y),
            'bounds': (xs.min(), ys.min(), xs.max(), ys.max())
        })
        print(f"  Component {i}: {size} pixels at ({center_x:.1f}, {center_y:.1f})")

# Sort by size
components.sort(key=lambda x: x['size'], reverse=True)

if len(components) > 1:
    main_comp = components[0]
    print(f"\nMain component: {main_comp['size']} pixels")
    print(f"Small components to remove:")
    
    for comp in components[1:]:
        # Calculate distance from main component
        main_cx, main_cy = main_comp['center']
        comp_cx, comp_cy = comp['center']
        distance = np.sqrt((main_cx - comp_cx)**2 + (main_cy - comp_cy)**2)
        
        print(f"  Component {comp['id']}: {comp['size']} pixels, "
              f"{distance:.1f} pixels from main, "
              f"at position ({comp_cx:.1f}, {comp_cy:.1f})")

# Create visualization
fig_height = 10
fig_width = 15

# Create output image
output = np.zeros((fig_height * 100, fig_width * 100, 3), dtype=np.uint8)

# Draw original sprite (top left)
if w_sprite.shape[0] < 300 and w_sprite.shape[1] < 400:
    orig_rgb = cv2.cvtColor(w_sprite, cv2.COLOR_RGBA2RGB)
    h, w = orig_rgb.shape[:2]
    output[50:50+h, 50:50+w] = orig_rgb
    cv2.putText(output, "Original W sprite", (50, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

# Draw each component in different colors (top right)
if w_sprite.shape[0] < 300 and w_sprite.shape[1] < 400:
    comp_colored = np.zeros((*w_sprite.shape[:2], 3), dtype=np.uint8)
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), 
              (255, 0, 255), (0, 255, 255), (128, 128, 255)]
    
    for i, comp in enumerate(components[:7]):  # Show up to 7 components
        mask = labeled == comp['id']
        comp_colored[mask] = colors[i % len(colors)]
    
    h, w = comp_colored.shape[:2]
    output[50:50+h, 500:500+w] = comp_colored
    cv2.putText(output, f"Components ({num_features} total)", (500, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

# Draw main component only (bottom left)
if len(components) > 0 and w_sprite.shape[0] < 300 and w_sprite.shape[1] < 400:
    main_only = w_sprite.copy()
    main_only[:, :, 3] = np.where(labeled == components[0]['id'], 
                                   main_only[:, :, 3], 0)
    main_rgb = cv2.cvtColor(main_only, cv2.COLOR_RGBA2RGB)
    h, w = main_rgb.shape[:2]
    output[400:400+h, 50:50+w] = main_rgb
    cv2.putText(output, "After removing small components", (50, 380),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

# Draw removed components (bottom right)
if len(components) > 1 and w_sprite.shape[0] < 300 and w_sprite.shape[1] < 400:
    removed_only = np.zeros((*w_sprite.shape[:2], 3), dtype=np.uint8)
    for comp in components[1:]:
        mask = labeled == comp['id']
        removed_only[mask] = (255, 0, 0)  # Red for removed pixels
    
    h, w = removed_only.shape[:2]
    output[400:400+h, 500:500+w] = removed_only
    cv2.putText(output, "Removed pixels (artifacts)", (500, 380),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

cv2.imwrite("w_sprite_components.png", output)
print(f"\nSaved visualization to w_sprite_components.png")

# Also save individual components for inspection
os.makedirs("w_components", exist_ok=True)
for i, comp in enumerate(components[:5]):  # Save up to 5 components
    comp_mask = labeled == comp['id']
    comp_img = np.zeros((*w_sprite.shape[:2], 4), dtype=np.uint8)
    comp_img[comp_mask] = w_sprite[comp_mask]
    
    # Crop to component bounds
    x1, y1, x2, y2 = comp['bounds']
    cropped = comp_img[y1:y2+1, x1:x2+1]
    
    # Scale up for visibility if very small
    if cropped.shape[0] < 20 or cropped.shape[1] < 20:
        scale = 10
        cropped = cv2.resize(cropped, None, fx=scale, fy=scale, 
                            interpolation=cv2.INTER_NEAREST)
    
    cv2.imwrite(f"w_components/component_{i}_{comp['size']}px.png", cropped)

print("Saved individual components to w_components/")