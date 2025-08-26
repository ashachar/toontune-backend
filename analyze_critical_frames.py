#!/usr/bin/env python3
"""Analyze the critical frames where occlusion should be most visible."""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os

print("🔬 Analyzing critical frames during dissolve phase...\n")

# Frames to check - during early dissolve when letters are still mostly opaque
frames_to_check = [20, 22, 24, 26, 28]

# Create comparison grid
grid_width = 640
grid_height = 360
grid = Image.new('RGB', (grid_width * 2, grid_height * len(frames_to_check)))

for row, frame_num in enumerate(frames_to_check):
    print(f"\n{'='*50}")
    print(f"Frame {frame_num} Analysis:")
    print('='*50)
    
    for col, (prefix, label) in enumerate([('broken', 'BEFORE FIX'), ('fixed', 'AFTER FIX')]):
        img_path = f'outputs/{prefix}_frame_{frame_num}.png'
        
        if not os.path.exists(img_path):
            print(f"  {label}: File not found")
            continue
            
        # Load and analyze
        img = Image.open(img_path)
        img_array = np.array(img)
        
        # Detect yellow letters
        yellow_mask = (
            (img_array[:, :, 0] > 180) & 
            (img_array[:, :, 1] > 180) & 
            (img_array[:, :, 2] < 120)
        )
        
        # Get foreground mask
        try:
            import sys
            sys.path.append('utils')
            from video.segmentation.segment_extractor import extract_foreground_mask
            
            mask = extract_foreground_mask(img_array)
            
            # Find overlap
            overlap = yellow_mask & (mask > 128)
            overlap_count = np.sum(overlap)
            
            # Focus on specific regions where we expect occlusion
            h, w = img_array.shape[:2]
            
            # Left region (where H and e are)
            left_third = w // 3
            left_yellow = np.sum(yellow_mask[:, :left_third])
            left_mask = np.sum(mask[:, :left_third] > 128)
            left_overlap = np.sum(overlap[:, :left_third])
            
            print(f"\n  {label}:")
            print(f"    Total yellow pixels: {np.sum(yellow_mask):,}")
            print(f"    Total overlap pixels: {overlap_count:,}")
            print(f"    Left region: {left_yellow:,} yellow, {left_mask:,} mask, {left_overlap:,} overlap")
            
            # Create visualization
            vis = img.resize((grid_width, grid_height), Image.Resampling.LANCZOS)
            draw = ImageDraw.Draw(vis)
            
            # Overlay mask edges and overlap
            vis_array = np.array(vis)
            vis_h, vis_w = vis_array.shape[:2]
            
            # Resize masks to match visualization
            yellow_resized = cv2.resize(yellow_mask.astype(np.uint8), (vis_w, vis_h))
            mask_resized = cv2.resize(mask, (vis_w, vis_h))
            overlap_resized = cv2.resize(overlap.astype(np.uint8), (vis_w, vis_h))
            
            # Create colored overlay
            overlay_img = vis.copy()
            overlay_array = np.array(overlay_img)
            
            # Green for mask edge
            edges = cv2.Canny(mask_resized, 100, 200)
            overlay_array[edges > 0] = [0, 255, 0]
            
            # Red for problematic overlap
            if overlap_count > 50:
                overlay_array[overlap_resized > 0] = [255, 0, 0]
            
            overlay_img = Image.fromarray(overlay_array)
            vis = Image.blend(vis, overlay_img, 0.4)
            
            # Add annotations
            draw = ImageDraw.Draw(vis)
            draw.rectangle([(0, 0), (vis_w, 30)], fill=(0, 0, 0, 200))
            draw.text((5, 5), f"{label} - Frame {frame_num}", fill=(255, 255, 255))
            
            if overlap_count > 50:
                draw.rectangle([(5, 35), (200, 55)], fill=(255, 0, 0, 200))
                draw.text((10, 37), f"❌ {overlap_count} pixels wrong!", fill=(255, 255, 255))
            else:
                draw.rectangle([(5, 35), (200, 55)], fill=(0, 255, 0, 200))
                draw.text((10, 37), f"✅ Occlusion OK", fill=(255, 255, 255))
            
            # Paste into grid
            grid.paste(vis, (col * grid_width, row * grid_height))
            
        except Exception as e:
            print(f"    Error: {e}")

# Add grid lines and labels
draw = ImageDraw.Draw(grid)
draw.line([(grid_width, 0), (grid_width, grid.height)], fill=(255, 255, 255), width=2)

for i in range(1, len(frames_to_check)):
    y = i * grid_height
    draw.line([(0, y), (grid.width, y)], fill=(128, 128, 128), width=1)

# Save grid
grid.save('outputs/critical_frames_comparison.png')
print("\n✅ Saved comparison grid to: outputs/critical_frames_comparison.png")

# Also create a detailed view of frame 24 (middle of dissolve)
print("\n🔍 Creating detailed view of frame 24...")
broken_24 = Image.open('outputs/broken_frame_24.png')
fixed_24 = Image.open('outputs/fixed_frame_24.png')

detail = Image.new('RGB', (broken_24.width * 2, broken_24.height))
detail.paste(broken_24, (0, 0))
detail.paste(fixed_24, (broken_24.width, 0))

draw = ImageDraw.Draw(detail)
draw.line([(broken_24.width, 0), (broken_24.width, detail.height)], fill=(255, 255, 255), width=2)

# Add titles
draw.rectangle([(0, 0), (broken_24.width, 40)], fill=(0, 0, 0, 200))
draw.rectangle([(broken_24.width, 0), (detail.width, 40)], fill=(0, 0, 0, 200))
draw.text((20, 10), "BEFORE: Static Mask", fill=(255, 0, 0))
draw.text((broken_24.width + 20, 10), "AFTER: Dynamic Mask", fill=(0, 255, 0))

detail.save('outputs/frame_24_detailed_comparison.png')
print("✅ Saved detailed comparison to: outputs/frame_24_detailed_comparison.png")