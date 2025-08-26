#!/usr/bin/env python3
"""Check opacity values in extracted frames to find the blink."""

from PIL import Image
import numpy as np

# Check frames around the handoff
for i in range(1, 7):
    img = Image.open(f'outputs/frame_{i:03d}.png').convert('RGBA')
    arr = np.array(img)
    
    # Find the letter pixels (non-transparent)
    letter_mask = arr[:, :, 3] > 0
    
    if np.any(letter_mask):
        # Get average alpha of letter pixels
        letter_alphas = arr[letter_mask, 3]
        avg_alpha = np.mean(letter_alphas)
        max_alpha = np.max(letter_alphas)
        min_alpha = np.min(letter_alphas[letter_alphas > 0]) if np.any(letter_alphas > 0) else 0
        
        # Get position of letter (center of mass)
        y_coords, x_coords = np.where(letter_mask)
        if len(y_coords) > 0:
            center_y = int(np.mean(y_coords))
            center_x = int(np.mean(x_coords))
        else:
            center_y = center_x = 0
            
        frame_num = 16 + i  # Actual frame number in video
        print(f"Frame {frame_num}: avg_alpha={avg_alpha:.1f}, max={max_alpha}, min={min_alpha}, center=({center_x}, {center_y})")
        
        # Check for sudden changes
        if i > 1:
            if abs(avg_alpha - prev_avg) > 50:
                print(f"  ⚠️ BLINK DETECTED! Alpha jumped from {prev_avg:.1f} to {avg_alpha:.1f}")
        
        prev_avg = avg_alpha
    else:
        print(f"Frame {16+i}: No letter pixels found")