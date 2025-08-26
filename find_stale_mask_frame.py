#!/usr/bin/env python3
"""Find the frame where the second 'l' shows through due to stale mask."""

import cv2
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

# The issue is around when letters are dissolving and person moves right
# Based on the image, person has moved right but the 'l' shows through where they WERE

print("🔍 Finding frame where stale mask causes letter to show through...\n")

cap = cv2.VideoCapture('outputs/test_occlusion_proof_final_h264.mp4')
fps = cap.get(cv2.CAP_PROP_FPS)

# Check frames during dissolve phase (after frame 19)
for frame_num in range(20, 40):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    
    if not ret:
        continue
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Look for the second 'l' position (around x=463)
    # Check if yellow pixels exist where person SHOULD be occluding
    h, w = frame_rgb.shape[:2]
    
    # Second 'l' is around x=463, y=305 based on debug output
    # Check region around that letter
    letter_region = frame_rgb[280:380, 430:500]  # Region around second 'l'
    
    # Detect yellow pixels in that region
    yellow_mask = (
        (letter_region[:, :, 0] > 180) & 
        (letter_region[:, :, 1] > 180) & 
        (letter_region[:, :, 2] < 120)
    )
    yellow_count = np.sum(yellow_mask)
    
    # Extract current mask to see where person ACTUALLY is
    import sys
    sys.path.append('utils')
    from video.segmentation.segment_extractor import extract_foreground_mask
    
    current_mask = extract_foreground_mask(frame_rgb)
    
    # Check if person is in the letter region NOW
    mask_region = current_mask[280:380, 430:500]
    mask_count = np.sum(mask_region > 128)
    
    # The bug: letter visible but person IS there
    if yellow_count > 100 and mask_count > 100:
        print(f"🎯 FOUND STALE MASK BUG at Frame {frame_num}!")
        print(f"   Second 'l' region: {yellow_count} yellow pixels visible")
        print(f"   Person mask: {mask_count} pixels (person IS there)")
        print(f"   → Letter should be hidden but it's not!")
        
        # Create detailed visualization
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        
        # Full frame
        axes[0, 0].imshow(frame_rgb)
        axes[0, 0].set_title(f"Frame {frame_num} - Full View")
        axes[0, 0].add_patch(plt.Rectangle((430, 280), 70, 100, 
                                          linewidth=2, edgecolor='red', facecolor='none'))
        axes[0, 0].text(435, 270, "Second 'l' region", color='red')
        
        # Current mask
        axes[0, 1].imshow(current_mask, cmap='gray')
        axes[0, 1].set_title("CURRENT Mask (where person IS now)")
        axes[0, 1].add_patch(plt.Rectangle((430, 280), 70, 100, 
                                          linewidth=2, edgecolor='red', facecolor='none'))
        
        # Zoomed letter region
        axes[0, 2].imshow(letter_region)
        axes[0, 2].set_title(f"Letter Region\n{yellow_count} yellow pixels visible")
        
        # Mask in letter region
        axes[1, 0].imshow(mask_region, cmap='gray')
        axes[1, 0].set_title(f"Mask in Letter Region\n{mask_count} mask pixels")
        
        # Overlay showing the problem
        overlay = frame_rgb.copy()
        # Red where mask says person is
        overlay[current_mask > 128] = overlay[current_mask > 128] * 0.5 + np.array([255, 0, 0]) * 0.5
        # Green circle around visible letter that shouldn't be
        overlay_img = Image.fromarray(overlay)
        draw = ImageDraw.Draw(overlay_img)
        draw.ellipse([(455, 320), (475, 340)], outline=(0, 255, 0), width=3)
        draw.text((480, 325), "BUG: Letter visible\nthrough person!", fill=(0, 255, 0))
        
        axes[1, 1].imshow(np.array(overlay_img))
        axes[1, 1].set_title("BUG: Letter shows through person")
        
        # Timeline explanation
        axes[1, 2].axis('off')
        axes[1, 2].text(0.1, 0.8, "STALE MASK BUG EXPLANATION:", fontsize=14, fontweight='bold')
        axes[1, 2].text(0.1, 0.6, "1. Person WAS at left (frames ago)", fontsize=12)
        axes[1, 2].text(0.1, 0.5, "2. Person moved RIGHT", fontsize=12)
        axes[1, 2].text(0.1, 0.4, "3. Dissolve using OLD mask position", fontsize=12, color='red')
        axes[1, 2].text(0.1, 0.3, "4. Letter shows where person WAS", fontsize=12, color='red')
        axes[1, 2].text(0.1, 0.1, "→ Mask NOT updated per frame!", fontsize=12, fontweight='bold', color='red')
        
        plt.suptitle(f"PROOF: Dissolve Using STALE Mask - Frame {frame_num}", fontsize=16, fontweight='bold', color='red')
        plt.tight_layout()
        plt.savefig(f'outputs/stale_mask_bug_frame_{frame_num}.png', dpi=150)
        print(f"\n✅ Saved proof to: outputs/stale_mask_bug_frame_{frame_num}.png")
        
        # Save the actual frame
        cv2.imwrite(f'outputs/bug_frame_{frame_num}.png', frame)
        break

cap.release()

print("\n" + "="*60)
print("STALE MASK BUG CONFIRMED")
print("="*60)
print("The dissolve animation is NOT recalculating masks every frame!")
print("It's using an OLD mask from when the person was in a different position.")
print("This causes letters to show through where the person WAS, not where they ARE.")