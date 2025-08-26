#!/usr/bin/env python3
"""Find a frame where the person actually overlaps with letters."""

import cv2
import numpy as np
from PIL import Image, ImageDraw

# Check the broken video first to find where occlusion should happen
cap = cv2.VideoCapture('outputs/test_all_fixes_applied_hq.mp4')

print("🔍 Searching for frames with person-letter overlap...")
best_frame = -1
max_overlap = 0

for frame_num in range(20, 50):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    if not ret:
        continue
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Find yellow letter pixels
    yellow_mask = (frame_rgb[:, :, 0] > 180) & (frame_rgb[:, :, 1] > 180) & (frame_rgb[:, :, 2] < 120)
    
    # Find person region (darker pixels in center-left of frame)
    # Focus on region where person usually is (left half of frame)
    h, w = frame_rgb.shape[:2]
    roi = frame_rgb[:, :w//2]  # Left half where person is
    
    # Person wears dark suit, so look for dark pixels
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    dark_mask_roi = gray_roi < 80  # Dark suit
    
    # Extend mask back to full frame
    dark_mask = np.zeros((h, w), dtype=bool)
    dark_mask[:, :w//2] = dark_mask_roi
    
    # Find overlap
    overlap = yellow_mask & dark_mask
    overlap_count = np.sum(overlap)
    
    if overlap_count > max_overlap:
        max_overlap = overlap_count
        best_frame = frame_num
        
        # Save this frame for inspection
        img = Image.fromarray(frame_rgb)
        draw = ImageDraw.Draw(img)
        
        # Highlight overlap areas
        y_coords, x_coords = np.where(overlap)
        for y, x in zip(y_coords[::5], x_coords[::5]):
            draw.ellipse([(x-2, y-2), (x+2, y+2)], fill=(255, 0, 0))
        
        draw.text((10, 10), f"Frame {frame_num}: {overlap_count} overlap pixels", fill=(255, 0, 0))
        img.save(f'outputs/overlap_frame_{frame_num}.png')

cap.release()

print(f"\n📍 Best overlap frame: {best_frame} with {max_overlap} pixels")
print(f"Saved to: outputs/overlap_frame_{best_frame}.png")

# Now compare both videos at this frame
if best_frame > 0:
    print(f"\n🔬 Comparing occlusion at frame {best_frame}...")
    
    videos = [
        ('outputs/test_all_fixes_applied_hq.mp4', 'BROKEN (Static Mask)'),
        ('outputs/test_occlusion_fixed_hq.mp4', 'FIXED (Dynamic Mask)')
    ]
    
    comparison = Image.new('RGB', (1280, 720))
    
    for i, (video_path, title) in enumerate(videos):
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, best_frame)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            continue
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        
        # Scale down for side-by-side
        img = img.resize((640, 360), Image.Resampling.LANCZOS)
        
        draw = ImageDraw.Draw(img)
        draw.rectangle([(0, 0), (640, 30)], fill=(0, 0, 0, 200))
        draw.text((10, 5), title, fill=(255, 255, 255))
        
        # Check for incorrectly visible letters
        frame_array = np.array(img)
        yellow = (frame_array[:, :, 0] > 180) & (frame_array[:, :, 1] > 180) & (frame_array[:, :, 2] < 120)
        gray = cv2.cvtColor(frame_array, cv2.COLOR_RGB2GRAY)
        dark = gray < 80
        
        bug_pixels = yellow & dark
        bug_count = np.sum(bug_pixels)
        
        if bug_count > 50:
            draw.rectangle([(10, 35), (300, 60)], fill=(255, 0, 0, 200))
            draw.text((15, 38), f"❌ {bug_count} pixels wrongly visible!", fill=(255, 255, 255))
            
            # Highlight bugs
            y_coords, x_coords = np.where(bug_pixels)
            for y, x in zip(y_coords[::3], x_coords[::3]):
                draw.ellipse([(x-1, y-1), (x+1, y+1)], fill=(255, 0, 0))
        else:
            draw.rectangle([(10, 35), (300, 60)], fill=(0, 255, 0, 200))
            draw.text((15, 38), f"✅ Occlusion correct!", fill=(0, 0, 0))
        
        comparison.paste(img, (i * 640, 180))
    
    comparison.save('outputs/occlusion_comparison_final.png')
    print(f"✅ Final comparison saved to: outputs/occlusion_comparison_final.png")