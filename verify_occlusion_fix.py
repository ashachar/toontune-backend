#!/usr/bin/env python3
"""Verify that the occlusion fix is working properly."""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Extract same frame from both videos
frame_to_check = 35

videos = [
    ('outputs/test_all_fixes_applied_hq.mp4', 'BEFORE FIX (Broken Occlusion)'),
    ('outputs/test_occlusion_fixed_hq.mp4', 'AFTER FIX (Dynamic Masking)')
]

comparison = None
for i, (video_path, title) in enumerate(videos):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_to_check)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print(f"Could not read frame from {video_path}")
        continue
    
    # Convert to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame_rgb)
    
    # Add title
    draw = ImageDraw.Draw(img)
    draw.rectangle([(0, 0), (img.width, 40)], fill=(0, 0, 0, 200))
    draw.text((10, 10), title, fill=(255, 255, 255))
    
    # Detect yellow pixels over dark areas (the bug)
    frame_array = np.array(img)
    yellow_mask = (frame_array[:, :, 0] > 200) & (frame_array[:, :, 1] > 200) & (frame_array[:, :, 2] < 100)
    gray = cv2.cvtColor(frame_array, cv2.COLOR_RGB2GRAY)
    person_mask = gray < 100
    bug_pixels = yellow_mask & person_mask
    bug_count = np.sum(bug_pixels)
    
    # Add bug count
    if bug_count > 100:
        draw.rectangle([(10, 50), (350, 80)], fill=(255, 0, 0, 200))
        draw.text((20, 55), f"⚠️ {bug_count} pixels incorrectly visible!", fill=(255, 255, 255))
        
        # Highlight problem areas
        y_coords, x_coords = np.where(bug_pixels)
        if len(y_coords) > 0:
            # Create red overlay for bug pixels
            overlay = img.copy()
            overlay_draw = ImageDraw.Draw(overlay)
            for y, x in zip(y_coords[::10], x_coords[::10]):  # Sample every 10th pixel
                overlay_draw.ellipse([(x-1, y-1), (x+1, y+1)], fill=(255, 0, 0))
            img = Image.blend(img, overlay, 0.5)
    else:
        draw.rectangle([(10, 50), (350, 80)], fill=(0, 128, 0, 200))
        draw.text((20, 55), f"✅ Occlusion working! ({bug_count} stray pixels)", fill=(255, 255, 255))
    
    # Extract the person's position for reference
    try:
        import sys
        sys.path.append('utils')
        from video.segmentation.segment_extractor import extract_foreground_mask
        
        mask = extract_foreground_mask(frame_array)
        
        # Overlay mask edge
        edges = cv2.Canny(mask, 100, 200)
        edge_overlay = np.zeros_like(frame_array)
        edge_overlay[:, :, 1] = edges  # Green channel for edges
        
        img_with_edges = Image.blend(img, Image.fromarray(edge_overlay), 0.3)
        img = img_with_edges
    except:
        pass
    
    if comparison is None:
        comparison = Image.new('RGB', (img.width * 2, img.height))
    
    comparison.paste(img, (i * img.width, 0))

# Draw dividing line
if comparison:
    draw = ImageDraw.Draw(comparison)
    draw.line([(comparison.width // 2, 0), (comparison.width // 2, comparison.height)], 
              fill=(255, 255, 255), width=2)
    
    comparison.save('outputs/occlusion_fix_verification.png')
    print(f"✅ Saved comparison to outputs/occlusion_fix_verification.png")
    print(f"Frame {frame_to_check} analyzed - check the image to see the improvement!")

# Also check multiple frames for statistics
print("\n📊 Checking occlusion quality across multiple frames...")
for video_path, title in videos:
    cap = cv2.VideoCapture(video_path)
    bug_counts = []
    
    for check_frame in range(20, 45, 5):
        cap.set(cv2.CAP_PROP_POS_FRAMES, check_frame)
        ret, frame = cap.read()
        if not ret:
            continue
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        yellow_mask = (frame_rgb[:, :, 0] > 200) & (frame_rgb[:, :, 1] > 200) & (frame_rgb[:, :, 2] < 100)
        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
        person_mask = gray < 100
        bug_pixels = yellow_mask & person_mask
        bug_counts.append(np.sum(bug_pixels))
    
    cap.release()
    
    if bug_counts:
        avg_bugs = np.mean(bug_counts)
        max_bugs = max(bug_counts)
        print(f"\n{title}:")
        print(f"  Average bug pixels: {avg_bugs:.0f}")
        print(f"  Max bug pixels: {max_bugs}")
        print(f"  Quality: {'❌ POOR' if avg_bugs > 500 else '⚠️ OK' if avg_bugs > 100 else '✅ GOOD'}")