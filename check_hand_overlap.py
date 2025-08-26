#!/usr/bin/env python3
"""Check for overlap in the specific hand/arm region where the bug was visible."""

import cv2
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

# The bug was visible around frame 25 (1 second) where H overlaps with left hand
print("🔍 Checking for hand/arm overlap with letters...\n")

# Check frames around 1 second mark
for frame_num in [23, 24, 25, 26, 27]:
    print(f"\nFrame {frame_num}:")
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    for idx, (video, title) in enumerate([
        ('outputs/test_all_fixes_applied_hq.mp4', 'BEFORE FIX'),
        ('outputs/test_occlusion_fixed_hq.mp4', 'AFTER FIX')
    ]):
        cap = cv2.VideoCapture(video)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            continue
            
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame_rgb.shape[:2]
        
        # Focus on the specific region where hands are
        # Based on the image you showed, hands are around y=300-500, x=100-400
        hand_region = frame_rgb[300:500, 100:400]
        
        # Detect yellow in this region
        yellow_mask = (
            (hand_region[:, :, 0] > 180) & 
            (hand_region[:, :, 1] > 180) & 
            (hand_region[:, :, 2] < 120)
        )
        
        yellow_count = np.sum(yellow_mask)
        
        # Check for dark pixels (suit/hands) in same region
        gray_region = cv2.cvtColor(hand_region, cv2.COLOR_RGB2GRAY)
        dark_mask = gray_region < 100
        dark_count = np.sum(dark_mask)
        
        # Find actual overlap
        overlap = yellow_mask & dark_mask
        overlap_count = np.sum(overlap)
        
        print(f"  {title}:")
        print(f"    Hand region: {yellow_count} yellow, {dark_count} dark, {overlap_count} overlap")
        
        # Visualize
        ax = axes[idx]
        
        # Show full frame
        ax.imshow(frame_rgb)
        
        # Draw rectangle around hand region
        rect = plt.Rectangle((100, 300), 300, 200, linewidth=2, 
                            edgecolor='red' if overlap_count > 0 else 'green', 
                            facecolor='none')
        ax.add_patch(rect)
        
        # Title
        ax.set_title(f"{title} - Frame {frame_num}\nOverlap: {overlap_count} pixels")
        ax.axis('off')
        
        # If there's overlap, highlight it
        if overlap_count > 0:
            # Create overlay
            overlay = np.zeros_like(frame_rgb)
            overlay[300:500, 100:400][overlap] = [255, 0, 0]
            
            # Blend
            frame_with_overlay = cv2.addWeighted(frame_rgb, 0.7, overlay, 0.3, 0)
            ax.imshow(frame_with_overlay)
    
    plt.tight_layout()
    plt.savefig(f'outputs/hand_overlap_frame_{frame_num}.png', dpi=100, bbox_inches='tight')
    plt.close()

print("\n✅ Saved hand overlap analysis to outputs/hand_overlap_frame_*.png")

# Create summary comparison
print("\n📊 Creating summary comparison...")
summary = Image.new('RGB', (1280, 720), (255, 255, 255))
draw = ImageDraw.Draw(summary)

# Load frame 25 from both videos
for i, (prefix, label, color) in enumerate([
    ('test_all_fixes_applied_hq', 'BEFORE (Static Mask)', (255, 0, 0)),
    ('test_occlusion_fixed_hq', 'AFTER (Dynamic Mask)', (0, 255, 0))
]):
    cap = cv2.VideoCapture(f'outputs/{prefix}.mp4')
    cap.set(cv2.CAP_PROP_POS_FRAMES, 25)
    ret, frame = cap.read()
    cap.release()
    
    if ret:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Crop to show just the relevant area
        cropped = frame_rgb[200:600, 50:500]
        img = Image.fromarray(cropped)
        img = img.resize((450, 320), Image.Resampling.LANCZOS)
        
        # Paste into summary
        x_offset = 50 + i * 580
        summary.paste(img, (x_offset, 150))
        
        # Add label
        draw.rectangle([(x_offset, 150), (x_offset + 450, 190)], fill=color)
        draw.text((x_offset + 10, 160), label, fill=(255, 255, 255))
        
        # Add analysis box
        draw.rectangle([(x_offset, 480), (x_offset + 450, 550)], outline=color, width=2)

# Title
draw.text((400, 50), "OCCLUSION FIX VERIFICATION", fill=(0, 0, 0))
draw.text((350, 80), "Frame 25 - Hand/Letter Overlap Region", fill=(64, 64, 64))

# Explanation
draw.text((50, 560), "Key: The fix ensures masks are extracted fresh every frame,", fill=(0, 0, 0))
draw.text((50, 580), "preventing letters from showing through moving body parts.", fill=(0, 0, 0))

summary.save('outputs/occlusion_fix_summary.png')
print("✅ Saved summary to outputs/occlusion_fix_summary.png")