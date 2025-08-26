#!/usr/bin/env python3
"""Verify the occlusion fix across multiple relevant frames."""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os

# Key frames to check (around dissolve phase when person is moving)
check_frames = [25, 30, 35, 40]  # During dissolve animation

videos = [
    ('outputs/test_all_fixes_applied_hq.mp4', 'BEFORE FIX'),
    ('outputs/test_occlusion_fixed_hq.mp4', 'AFTER FIX')
]

# Create output directory
os.makedirs('outputs/verification_frames', exist_ok=True)

print("🔍 Extracting and analyzing multiple frames to verify dynamic masking...\n")

for video_path, video_label in videos:
    print(f"\n{'='*60}")
    print(f"Analyzing: {video_label}")
    print('='*60)
    
    cap = cv2.VideoCapture(video_path)
    
    for frame_num in check_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        
        if not ret:
            print(f"Frame {frame_num}: Could not read")
            continue
        
        # Convert to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame_rgb.shape[:2]
        
        # Detect yellow letters
        yellow_mask = (frame_rgb[:, :, 0] > 180) & (frame_rgb[:, :, 1] > 180) & (frame_rgb[:, :, 2] < 100)
        yellow_pixels = np.sum(yellow_mask)
        
        # Extract foreground mask to see what SHOULD be occluding
        try:
            import sys
            sys.path.append('utils')
            from video.segmentation.segment_extractor import extract_foreground_mask
            
            current_mask = extract_foreground_mask(frame_rgb)
            foreground_pixels = np.sum(current_mask > 0)
            
            # Find overlap: letters that SHOULD be hidden
            overlap_mask = yellow_mask & (current_mask > 128)
            overlap_pixels = np.sum(overlap_mask)
            
            # Check specific regions where person typically overlaps with text
            # Left side where 'H' and 'e' might be
            left_region_yellow = np.sum(yellow_mask[:, :w//3])
            left_region_person = np.sum(current_mask[:, :w//3] > 128)
            
            # Center region
            center_region_yellow = np.sum(yellow_mask[:, w//3:2*w//3])
            center_region_person = np.sum(current_mask[:, w//3:2*w//3] > 128)
            
            print(f"\nFrame {frame_num}:")
            print(f"  Yellow pixels visible: {yellow_pixels:,}")
            print(f"  Foreground (person) pixels: {foreground_pixels:,}")
            print(f"  Overlap (should be hidden): {overlap_pixels:,} pixels")
            print(f"  Left region: {left_region_yellow:,} yellow, {left_region_person:,} person")
            print(f"  Center region: {center_region_yellow:,} yellow, {center_region_person:,} person")
            
            # Create annotated frame
            img = Image.fromarray(frame_rgb)
            draw = ImageDraw.Draw(img)
            
            # Show mask overlay
            mask_overlay = np.zeros_like(frame_rgb)
            mask_overlay[:, :, 1] = current_mask  # Green for mask
            mask_overlay[overlap_mask, 0] = 255  # Red for problematic overlap
            
            img_with_overlay = Image.blend(img, Image.fromarray(mask_overlay), 0.3)
            
            # Add text annotations
            draw = ImageDraw.Draw(img_with_overlay)
            draw.rectangle([(0, 0), (400, 60)], fill=(0, 0, 0, 200))
            draw.text((10, 5), f"{video_label} - Frame {frame_num}", fill=(255, 255, 255))
            draw.text((10, 25), f"Yellow pixels: {yellow_pixels:,}", fill=(255, 255, 0))
            draw.text((10, 40), f"Should hide: {overlap_pixels:,} pixels", 
                     fill=(255, 0, 0) if overlap_pixels > 100 else (0, 255, 0))
            
            # If there's significant overlap that should be hidden
            if overlap_pixels > 100:
                # Highlight problem areas
                y_coords, x_coords = np.where(overlap_mask)
                if len(y_coords) > 0:
                    # Draw circles around problem areas
                    for i in range(0, len(y_coords), 50):
                        draw.ellipse([(x_coords[i]-3, y_coords[i]-3), 
                                    (x_coords[i]+3, y_coords[i]+3)], 
                                    outline=(255, 0, 0), width=2)
                
                draw.text((10, 80), "⚠️ OCCLUSION BUG DETECTED!", fill=(255, 0, 0))
            else:
                draw.text((10, 80), "✅ Occlusion working correctly", fill=(0, 255, 0))
            
            # Save annotated frame
            output_path = f'outputs/verification_frames/{video_label.replace(" ", "_")}_frame_{frame_num}.png'
            img_with_overlay.save(output_path)
            
        except Exception as e:
            print(f"  Error extracting mask: {e}")
    
    cap.release()

# Create a grid comparison
print("\n📊 Creating grid comparison of all frames...")

grid = Image.new('RGB', (1280 * 2, 360 * len(check_frames)))

for i, (video_path, video_label) in enumerate(videos):
    for j, frame_num in enumerate(check_frames):
        img_path = f'outputs/verification_frames/{video_label.replace(" ", "_")}_frame_{frame_num}.png'
        if os.path.exists(img_path):
            img = Image.open(img_path)
            # Scale to fit
            img = img.resize((1280, 360), Image.Resampling.LANCZOS)
            grid.paste(img, (i * 1280, j * 360))

# Add labels
draw = ImageDraw.Draw(grid)
draw.line([(1280, 0), (1280, grid.height)], fill=(255, 255, 255), width=2)

for j, frame_num in enumerate(check_frames):
    draw.line([(0, (j+1)*360), (grid.width, (j+1)*360)], fill=(128, 128, 128), width=1)

grid.save('outputs/occlusion_verification_grid.png')
print("✅ Saved grid comparison to: outputs/occlusion_verification_grid.png")

# Summary statistics
print("\n" + "="*60)
print("VERIFICATION SUMMARY")
print("="*60)
print("\nThe AFTER FIX video should show:")
print("1. Consistent masking across all frames")
print("2. Lower or zero 'should hide' pixels")
print("3. Dynamic adjustment as person moves")
print("\nCheck the grid image to visually confirm the fix!")