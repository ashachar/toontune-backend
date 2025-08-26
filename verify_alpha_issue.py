#!/usr/bin/env python3
"""Verify that motion is returning RGB with alpha baked in."""

import numpy as np
from PIL import Image

# Simulate what motion does
def motion_behavior():
    # Create a white letter on transparent background
    sprite = np.zeros((100, 100, 4), dtype=np.uint8)
    sprite[30:70, 30:70] = [255, 255, 255, 255]  # White square with full alpha
    
    # Apply alpha (like line 324 in motion)
    alpha = 0.63
    sprite[:, :, 3] = (sprite[:, :, 3] * alpha).astype(np.uint8)
    print(f"After motion alpha: max_alpha={sprite[:,:,3].max()} (should be ~161)")
    
    # Create canvas and paste (like line 384)
    canvas = Image.new('RGBA', (200, 200), (0, 0, 0, 0))
    sprite_img = Image.fromarray(sprite)
    canvas.paste(sprite_img, (50, 50), sprite_img)
    
    # Return RGB only (like line 418)
    result = np.array(canvas)
    rgb_only = result[:, :, :3]
    
    # Check what we get
    letter_region = rgb_only[80:120, 80:120]
    print(f"RGB values in letter region: {letter_region[letter_region > 0][0]}")
    print("Motion returns RGB with alpha already applied via paste operation")
    
    return rgb_only, sprite_img

def dissolve_behavior(rgb_frame, sprite_from_motion):
    """Simulate what dissolve does with the handoff."""
    # Dissolve receives RGB frame and sprites
    print("\nDissolve receives:")
    print(f"- RGB frame (no alpha channel)")
    print(f"- Sprite with alpha={np.array(sprite_from_motion)[:,:,3].max()}")
    
    # Problem: dissolve applies another alpha multiplier!
    target_alpha = 0.63
    handoff_alpha = 0.63
    relative_mult = target_alpha / handoff_alpha  # = 1.0
    
    # But the sprite already has alpha baked in!
    sprite_arr = np.array(sprite_from_motion)
    print(f"Sprite alpha before multiplier: {sprite_arr[:,:,3].max()}")
    
    # This doesn't change it (mult=1.0) but the alpha is already reduced!
    sprite_arr[:, :, 3] = (sprite_arr[:, :, 3] * relative_mult).astype(np.uint8)
    print(f"Sprite alpha after multiplier: {sprite_arr[:,:,3].max()}")
    
    # When pasted, this already-reduced alpha creates the appearance

rgb_frame, sprite = motion_behavior()
dissolve_behavior(rgb_frame, sprite)