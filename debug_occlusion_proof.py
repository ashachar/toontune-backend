#!/usr/bin/env python3
"""Debug script to prove if masks are being extracted and applied."""

import sys
import os
sys.path.append('utils')

# Add debug logging to the dissolve animation
debug_code = '''
# OCCLUSION DEBUG: Prove mask extraction and application
def debug_occlusion(frame_num, mask, letter_pos, sprite_array, applied_result):
    """Log detailed occlusion information."""
    import numpy as np
    
    # Check if mask exists
    if mask is None:
        print(f"[OCCLUSION_DEBUG] Frame {frame_num}: NO MASK - letters will NOT be occluded!")
        return
    
    # Check mask properties
    mask_pixels = np.sum(mask > 0)
    print(f"[OCCLUSION_DEBUG] Frame {frame_num}: Mask has {mask_pixels:,} foreground pixels")
    
    # Check if letter position overlaps with mask
    x, y = letter_pos
    h, w = sprite_array.shape[:2]
    
    # Extract mask region where letter is
    mask_region = mask[y:y+h, x:x+w]
    occluding_pixels = np.sum(mask_region > 128)
    
    # Check sprite alpha before and after
    alpha_before = sprite_array[:, :, 3]
    non_zero_before = np.sum(alpha_before > 0)
    
    if applied_result is not None:
        alpha_after = applied_result[:, :, 3]
        non_zero_after = np.sum(alpha_after > 0)
        hidden_pixels = non_zero_before - non_zero_after
    else:
        non_zero_after = non_zero_before
        hidden_pixels = 0
    
    print(f"[OCCLUSION_DEBUG] Letter at ({x},{y}): {occluding_pixels} mask pixels, {non_zero_before} visible → {non_zero_after} ({hidden_pixels} hidden)")
'''

print("Debug code to add to letter_3d_dissolve.py:")
print(debug_code)

# Now let's check the actual code to see what's happening
from video.segmentation.segment_extractor import extract_foreground_mask
import cv2
import numpy as np

# Load a test frame
cap = cv2.VideoCapture('outputs/test_occlusion_fixed_h264.mp4')
cap.set(cv2.CAP_PROP_POS_FRAMES, 25)  # Frame where issue is visible
ret, frame = cap.read()
cap.release()

if ret:
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    print("\n" + "="*60)
    print("TESTING MASK EXTRACTION ON FRAME 25:")
    print("="*60)
    
    # Extract mask
    mask = extract_foreground_mask(frame_rgb)
    
    print(f"✓ Mask extracted: shape={mask.shape}, dtype={mask.dtype}")
    print(f"✓ Foreground pixels: {np.sum(mask > 0):,}")
    print(f"✓ Max value: {mask.max()}, Min value: {mask.min()}")
    
    # Check specific region where 'H' letter should be
    # Based on the image, H is around x=75-175, y=260-380
    h_region = mask[260:380, 75:175]
    h_mask_pixels = np.sum(h_region > 128)
    
    print(f"\n'H' letter region (75-175, 260-380):")
    print(f"  Mask pixels in region: {h_mask_pixels}")
    print(f"  Should occlude: {'YES' if h_mask_pixels > 100 else 'NO'}")
    
    # Save visualization
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Original frame
    axes[0,0].imshow(frame_rgb)
    axes[0,0].set_title("Original Frame 25")
    axes[0,0].add_patch(plt.Rectangle((75, 260), 100, 120, 
                                      linewidth=2, edgecolor='red', facecolor='none'))
    axes[0,0].text(75, 250, "'H' region", color='red')
    
    # Mask
    axes[0,1].imshow(mask, cmap='gray')
    axes[0,1].set_title(f"Extracted Mask ({np.sum(mask>0):,} pixels)")
    axes[0,1].add_patch(plt.Rectangle((75, 260), 100, 120, 
                                      linewidth=2, edgecolor='red', facecolor='none'))
    
    # Mask in H region
    axes[1,0].imshow(h_region, cmap='gray')
    axes[1,0].set_title(f"Mask in 'H' Region ({h_mask_pixels} pixels)")
    
    # Overlay
    overlay = frame_rgb.copy()
    overlay[mask > 128] = [255, 0, 0]  # Red where mask is
    axes[1,1].imshow(overlay)
    axes[1,1].set_title("Mask Overlay (Red = Should Hide)")
    
    plt.tight_layout()
    plt.savefig('outputs/mask_extraction_proof.png', dpi=150)
    print("\n✓ Saved visualization to outputs/mask_extraction_proof.png")
    plt.close()
    
    # Now check if the masking logic in dissolve is actually being applied
    print("\n" + "="*60)
    print("CHECKING DISSOLVE MASKING LOGIC:")
    print("="*60)
    
    # Simulate what dissolve does
    print("\nWhat dissolve SHOULD do when is_behind=True:")
    print("1. Extract mask from current frame ✓")
    print("2. For each letter sprite:")
    print("   a. Get mask region at letter position")
    print("   b. Multiply sprite alpha by (1 - mask/255)")
    print("   c. This makes pixels transparent where mask > 0")
    print("\nLet's check if this is actually happening...")