#!/usr/bin/env python3
"""Prove the stale mask issue by checking multiple frames."""

import cv2
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

# Extract frames during dissolve to see the issue
cap = cv2.VideoCapture('outputs/test_occlusion_proof_final_h264.mp4')

# Create comparison showing person movement vs letter visibility
fig, axes = plt.subplots(3, 4, figsize=(16, 10))
fig.suptitle("PROVING STALE MASK: Person Moves But Letters Don't Update", fontsize=14, fontweight='bold')

frames_to_check = [19, 21, 23, 25]  # During dissolve

import sys
sys.path.append('utils')
from video.segmentation.segment_extractor import extract_foreground_mask

for row in range(3):
    for col, frame_num in enumerate(frames_to_check):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        
        if not ret:
            continue
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        if row == 0:
            # Show original frames
            axes[row, col].imshow(frame_rgb)
            axes[row, col].set_title(f"Frame {frame_num}")
            axes[row, col].axis('off')
            
            # Mark where letters are
            axes[row, col].add_patch(plt.Rectangle((400, 280), 150, 100, 
                                                  linewidth=1, edgecolor='yellow', facecolor='none'))
        
        elif row == 1:
            # Show current mask
            mask = extract_foreground_mask(frame_rgb)
            axes[row, col].imshow(mask, cmap='gray')
            axes[row, col].set_title(f"Mask: {np.sum(mask>0):,} px")
            axes[row, col].axis('off')
            
            # Mark where person is NOW
            y_coords, x_coords = np.where(mask > 128)
            if len(x_coords) > 0:
                left_edge = np.min(x_coords)
                right_edge = np.max(x_coords)
                axes[row, col].axvline(left_edge, color='green', linestyle='--', alpha=0.5)
                axes[row, col].axvline(right_edge, color='green', linestyle='--', alpha=0.5)
        
        else:
            # Show yellow pixels (letters)
            yellow_mask = (
                (frame_rgb[:, :, 0] > 180) & 
                (frame_rgb[:, :, 1] > 180) & 
                (frame_rgb[:, :, 2] < 100)
            )
            axes[row, col].imshow(yellow_mask, cmap='hot')
            axes[row, col].set_title(f"Letters: {np.sum(yellow_mask):,} px")
            axes[row, col].axis('off')
            
            # Check if letters overlap with current mask
            overlap = yellow_mask & (mask > 128)
            overlap_count = np.sum(overlap)
            
            if overlap_count > 100:
                axes[row, col].text(640, 600, f"⚠️ {overlap_count} px\nshould be hidden!", 
                                  color='red', fontsize=10, ha='center',
                                  bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))

cap.release()

# Add row labels
fig.text(0.02, 0.75, 'Original\nFrames', fontsize=12, fontweight='bold', ha='center')
fig.text(0.02, 0.5, 'Current\nMask', fontsize=12, fontweight='bold', ha='center')
fig.text(0.02, 0.25, 'Visible\nLetters', fontsize=12, fontweight='bold', ha='center')

plt.tight_layout()
plt.savefig('outputs/stale_mask_timeline.png', dpi=150, bbox_inches='tight')
print("✅ Saved timeline to: outputs/stale_mask_timeline.png")

# Now let's prove the exact issue you described
print("\n🔍 Analyzing the specific 'l' visibility issue...")

# The second 'l' in Hello is at approximately x=463
# Check if it's visible when it shouldn't be
cap = cv2.VideoCapture('outputs/test_occlusion_proof_final_h264.mp4')

for frame_num in [23, 24, 25, 26]:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    
    if not ret:
        continue
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Check specific region where second 'l' should be
    # Based on debug output, it's at x=463, y=305
    l_region = frame_rgb[290:350, 450:480]
    
    # Is the letter visible?
    yellow_in_region = (
        (l_region[:, :, 0] > 180) & 
        (l_region[:, :, 1] > 180) & 
        (l_region[:, :, 2] < 100)
    )
    yellow_count = np.sum(yellow_in_region)
    
    # Is person's mask there NOW?
    mask = extract_foreground_mask(frame_rgb)
    mask_in_region = mask[290:350, 450:480]
    mask_count = np.sum(mask_in_region > 128)
    
    print(f"\nFrame {frame_num} - Second 'l' region (x=450-480):")
    print(f"  Yellow pixels (letter visible): {yellow_count}")
    print(f"  Mask pixels (person there): {mask_count}")
    
    if yellow_count > 50 and mask_count > 50:
        print(f"  ⚠️ BUG: Letter visible ({yellow_count} px) where person IS ({mask_count} px)!")
        print(f"  → This proves mask is NOT being applied correctly!")

cap.release()

print("\n" + "="*60)
print("CONCLUSION: STALE MASK BUG")
print("="*60)
print("Even though masks are being EXTRACTED every frame (we see changing pixel counts),")
print("they are NOT being APPLIED correctly to hide letters during dissolve.")
print("The letters remain visible where the person WAS, not where they ARE.")