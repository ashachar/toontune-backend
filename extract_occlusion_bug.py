#!/usr/bin/env python3
"""Extract and highlight the foreground occlusion bug."""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Extract frame from the video where the bug is visible
cap = cv2.VideoCapture('outputs/test_all_fixes_applied_hq.mp4')

# Jump to around frame 30-40 where dissolve is happening and person might be moving
target_frame = 35
cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
ret, frame = cap.read()

if not ret:
    print("Could not read frame")
    exit(1)

# Convert to RGB
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
img = Image.fromarray(frame_rgb)

# Save original
img.save('outputs/occlusion_bug_frame.png')
print(f"✅ Saved frame {target_frame} to outputs/occlusion_bug_frame.png")

# Create annotation image
annotated = img.copy()
draw = ImageDraw.Draw(annotated)

# Try to detect where letters should be occluded
# Look for yellow pixels (letters) that appear over dark areas (person)
frame_array = np.array(img)

# Detect yellow letter pixels
yellow_mask = (frame_array[:, :, 0] > 200) & (frame_array[:, :, 1] > 200) & (frame_array[:, :, 2] < 100)

# Detect dark/person areas (low brightness)
gray = cv2.cvtColor(frame_array, cv2.COLOR_RGB2GRAY)
person_mask = gray < 100  # Dark areas likely to be person

# Find overlap - letters that SHOULD be hidden
bug_pixels = yellow_mask & person_mask
bug_locations = np.where(bug_pixels)

if len(bug_locations[0]) > 0:
    print(f"🐛 Found {len(bug_locations[0])} pixels where letters appear over foreground!")
    
    # Draw red circles around problem areas
    # Group nearby pixels into regions
    y_coords, x_coords = bug_locations
    
    # Find clusters of bug pixels
    clusters = []
    used = set()
    
    for i in range(len(y_coords)):
        if i in used:
            continue
        cluster_y = [y_coords[i]]
        cluster_x = [x_coords[i]]
        used.add(i)
        
        # Find nearby pixels
        for j in range(i+1, len(y_coords)):
            if j in used:
                continue
            if abs(y_coords[j] - np.mean(cluster_y)) < 30 and abs(x_coords[j] - np.mean(cluster_x)) < 30:
                cluster_y.append(y_coords[j])
                cluster_x.append(x_coords[j])
                used.add(j)
        
        if len(cluster_y) > 10:  # Only mark significant clusters
            clusters.append((int(np.mean(cluster_x)), int(np.mean(cluster_y)), len(cluster_y)))
    
    # Draw annotations
    for cx, cy, size in clusters[:5]:  # Top 5 problem areas
        # Draw red circle
        draw.ellipse([(cx-20, cy-20), (cx+20, cy+20)], outline=(255, 0, 0), width=3)
        # Draw arrow pointing to issue
        draw.line([(cx+30, cy-30), (cx+15, cy-15)], fill=(255, 0, 0), width=2)
        draw.text((cx+35, cy-35), "BUG!", fill=(255, 0, 0))
    
    # Add text explanation
    draw.rectangle([(10, 10), (600, 80)], fill=(0, 0, 0, 200))
    draw.text((20, 15), "🐛 OCCLUSION BUG DETECTED!", fill=(255, 0, 0))
    draw.text((20, 35), f"Frame {target_frame}: Letters visible through foreground", fill=(255, 255, 255))
    draw.text((20, 55), "Red circles = Letters that SHOULD be hidden by person", fill=(255, 200, 200))

# Also extract the foreground mask to show what SHOULD be hiding the letters
try:
    import sys
    import os
    sys.path.append('utils')
    from video.segmentation.segment_extractor import extract_foreground_mask
    
    mask = extract_foreground_mask(frame_array)
    
    # Create side-by-side comparison
    comparison = Image.new('RGB', (img.width * 2, img.height))
    comparison.paste(annotated, (0, 0))
    
    # Show mask overlay
    mask_colored = np.zeros_like(frame_array)
    mask_colored[:, :, 0] = mask  # Red channel for mask
    mask_overlay = Image.blend(img, Image.fromarray(mask_colored), 0.5)
    
    draw2 = ImageDraw.Draw(mask_overlay)
    draw2.text((20, 15), "Expected Foreground Mask", fill=(255, 255, 255))
    draw2.text((20, 35), "Red = Should hide letters", fill=(255, 200, 200))
    
    comparison.paste(mask_overlay, (img.width, 0))
    comparison.save('outputs/occlusion_bug_comparison.png')
    print("✅ Saved comparison to outputs/occlusion_bug_comparison.png")
    
except Exception as e:
    print(f"Could not extract mask: {e}")

annotated.save('outputs/occlusion_bug_annotated.png')
print("✅ Saved annotated frame to outputs/occlusion_bug_annotated.png")

# Try multiple frames to find the best example
print("\n🔍 Checking multiple frames for the clearest bug example...")
best_bug_count = 0
best_frame = target_frame

for check_frame in range(25, 45):
    cap.set(cv2.CAP_PROP_POS_FRAMES, check_frame)
    ret, check = cap.read()
    if not ret:
        continue
    
    check_rgb = cv2.cvtColor(check, cv2.COLOR_BGR2RGB)
    check_array = np.array(check_rgb)
    
    # Quick check for bug pixels
    yellow = (check_array[:, :, 0] > 200) & (check_array[:, :, 1] > 200) & (check_array[:, :, 2] < 100)
    gray = cv2.cvtColor(check_array, cv2.COLOR_RGB2GRAY)
    dark = gray < 100
    
    bug_count = np.sum(yellow & dark)
    if bug_count > best_bug_count:
        best_bug_count = bug_count
        best_frame = check_frame

if best_frame != target_frame:
    print(f"📍 Better example found at frame {best_frame} with {best_bug_count} bug pixels")
    cap.set(cv2.CAP_PROP_POS_FRAMES, best_frame)
    ret, best = cap.read()
    if ret:
        best_rgb = cv2.cvtColor(best, cv2.COLOR_BGR2RGB)
        Image.fromarray(best_rgb).save('outputs/occlusion_bug_best_frame.png')
        print(f"✅ Saved best bug example to outputs/occlusion_bug_best_frame.png")

cap.release()