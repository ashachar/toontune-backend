#!/usr/bin/env python3
"""Debug why W sprite doesn't have disconnected components removed."""

import os
os.environ['FRAME_DISSOLVE_DEBUG'] = '1'

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage
import cv2

# Simulate the exact conditions from WordDissolve._prepare_letter_sprites
def extract_w_sprite_like_worddissolve():
    # Parameters matching hello_world_fixed test
    word = "HELLO WORLD"
    font_size = 130  # From test file
    width, height = 1280, 492
    text_x, text_y = 50, 180  # Approximate
    
    # Load font
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", font_size)
    except:
        font = ImageFont.load_default()
    
    # Build per-letter masks using prefix difference (exact method from word_dissolve.py)
    masks_seed = []
    prefix = ""
    
    for ch in word:
        imgA = Image.new('L', (width, height), 0)
        imgB = Image.new('L', (width, height), 0)
        dA, dB = ImageDraw.Draw(imgA), ImageDraw.Draw(imgB)
        
        if prefix:
            dA.text((text_x, text_y), prefix, font=font, fill=255)
        dB.text((text_x, text_y), prefix + ch, font=font, fill=255)
        
        # Get difference
        arrA = np.array(imgA)
        arrB = np.array(imgB)
        
        # This is the seed mask for this character
        seed = arrB > arrA
        masks_seed.append(seed)
        prefix += ch
    
    # Now process the W (index 6)
    w_index = 6
    print(f"Processing character at index {w_index}: '{word[w_index]}'")
    
    # Get the W mask
    w_mask = masks_seed[w_index]
    
    # Find bounding box
    ys, xs = np.where(w_mask)
    if len(xs) == 0:
        print("No pixels found for W!")
        return
    
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    print(f"W bounding box: ({x_min}, {y_min}) to ({x_max}, {y_max})")
    
    # Create a full RGBA text image to extract sprite from
    full_img = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(full_img)
    draw.text((text_x, text_y), word, font=font, fill=(255, 220, 0, 255))  # Yellow text
    frozen_rgba = np.array(full_img)
    
    # Extract W sprite based on mask
    x0, y0, x1, y1 = x_min, y_min, x_max + 1, y_max + 1
    w_sprite = frozen_rgba[y0:y1, x0:x1].copy()
    
    # Mask out non-W pixels
    w_mask_cropped = w_mask[y0:y1, x0:x1]
    w_sprite[~w_mask_cropped] = 0
    
    print(f"W sprite shape: {w_sprite.shape}")
    print(f"Non-zero alpha pixels: {np.sum(w_sprite[:,:,3] > 0)}")
    
    # Now apply the same processing as word_dissolve.py
    alpha_threshold = 5
    w_sprite[:, :, 3] = np.where(w_sprite[:, :, 3] < alpha_threshold, 0, w_sprite[:, :, 3])
    
    # Check for connected components
    alpha_binary = w_sprite[:, :, 3] > alpha_threshold
    
    if np.any(alpha_binary):
        labeled, num_features = ndimage.label(alpha_binary)
        print(f"\nConnected components in W: {num_features}")
        
        if num_features > 1:
            # Analyze components
            sizes = []
            for i in range(1, num_features + 1):
                size = np.sum(labeled == i)
                sizes.append(size)
                
                # Find position of this component
                comp_ys, comp_xs = np.where(labeled == i)
                if len(comp_xs) > 0:
                    min_x, max_x = comp_xs.min(), comp_xs.max()
                    min_y, max_y = comp_ys.min(), comp_ys.max()
                    print(f"  Component {i}: {size} pixels, bounds: ({min_x},{min_y})-({max_x},{max_y})")
            
            # Which is largest?
            largest_idx = np.argmax(sizes) + 1
            print(f"\nLargest component: #{largest_idx} with {sizes[largest_idx-1]} pixels")
            print(f"Would remove {num_features - 1} components")
            
            # Visualize
            fig = np.zeros((w_sprite.shape[0], w_sprite.shape[1] * 3, 3), dtype=np.uint8)
            
            # Original
            fig[:, :w_sprite.shape[1]] = w_sprite[:, :, :3]
            
            # Components colored
            colors = [(255,0,0), (0,255,0), (0,0,255), (255,255,0), (255,0,255)]
            comp_vis = np.zeros((w_sprite.shape[0], w_sprite.shape[1], 3), dtype=np.uint8)
            for i in range(1, min(num_features + 1, 6)):
                mask = labeled == i
                comp_vis[mask] = colors[(i-1) % len(colors)]
            fig[:, w_sprite.shape[1]:w_sprite.shape[1]*2] = comp_vis
            
            # After cleanup
            cleaned = w_sprite.copy()
            cleaned[:, :, 3] = np.where(labeled == largest_idx, cleaned[:, :, 3], 0)
            fig[:, w_sprite.shape[1]*2:] = cleaned[:, :, :3]
            
            cv2.imwrite("w_components_debug.png", fig)
            print("\nSaved visualization to w_components_debug.png")
            print("Left: Original | Middle: Components (colored) | Right: After cleanup")
        else:
            print("Only 1 component found - no cleanup needed")
    else:
        print("No pixels above threshold!")
    
    # Also check what happens with the ACTUAL sprite from the video
    print("\n" + "="*50)
    print("Checking actual W from video frame...")
    
    # Extract W from actual rendered frame
    cap = cv2.VideoCapture("hello_world_fixed.mp4")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 95)  # Frame where W is visible
    ret, frame = cap.read()
    cap.release()
    
    if ret:
        # Extract W region
        w_region = frame[y_min:y_max+1, x_min:x_max+1]
        
        # Convert to RGBA
        w_rgba = cv2.cvtColor(w_region, cv2.COLOR_BGR2RGBA)
        
        # Create mask from yellow pixels
        yellow_mask = (w_rgba[:,:,1] > 180) & (w_rgba[:,:,2] > 180) & (w_rgba[:,:,0] < 100)
        w_rgba[~yellow_mask, 3] = 0
        
        # Check components
        alpha_binary = w_rgba[:,:,3] > 0
        if np.any(alpha_binary):
            labeled, num_features = ndimage.label(alpha_binary)
            print(f"Connected components in actual W from video: {num_features}")
            
            for i in range(1, min(num_features + 1, 6)):
                size = np.sum(labeled == i)
                print(f"  Component {i}: {size} pixels")

extract_w_sprite_like_worddissolve()