#!/usr/bin/env python3
"""Visualize the handoff issue between motion and dissolve."""

import numpy as np
from PIL import Image
import cv2

def simulate_handoff():
    """Simulate the exact handoff issue."""
    
    # Create a background
    background = np.full((200, 200, 3), 50, dtype=np.uint8)  # Dark gray
    
    # Create a white letter sprite
    letter = np.zeros((50, 50, 4), dtype=np.uint8)
    letter[10:40, 10:40] = [255, 255, 255, 255]  # White square
    
    print("=== MOTION ANIMATION (Frame 18) ===")
    # Motion applies alpha to sprite
    alpha = 0.63
    letter_motion = letter.copy()
    letter_motion[:, :, 3] = (letter_motion[:, :, 3] * alpha).astype(np.uint8)
    print(f"Motion sprite alpha: max={letter_motion[:,:,3].max()}")
    
    # Motion composites onto background
    canvas_motion = Image.fromarray(np.concatenate([background, np.zeros((200, 200, 1), dtype=np.uint8)], axis=2))
    letter_img = Image.fromarray(letter_motion)
    canvas_motion.paste(letter_img, (75, 75), letter_img)
    
    # Motion returns RGB only
    motion_result = np.array(canvas_motion)[:, :, :3]
    
    # Check the actual pixel values
    letter_region = motion_result[85:125, 85:125]
    letter_pixels = letter_region[letter_region > 50]
    if len(letter_pixels) > 0:
        print(f"Motion result: letter pixels = {letter_pixels[0]} (on background=50)")
        expected = int(50 + (255 - 50) * alpha)
        print(f"Expected: {expected} (50 + (255-50) * 0.63)")
    
    print("\n=== DISSOLVE ANIMATION (Frame 19) ===")
    # Dissolve starts with fresh background
    canvas_dissolve = Image.fromarray(np.concatenate([background, np.zeros((200, 200, 1), dtype=np.uint8)], axis=2))
    
    # Dissolve uses the sprites from motion (already at alpha=160)
    # It calculates relative multiplier = 0.63/0.63 = 1.0
    letter_dissolve = letter_motion.copy()  # Already has alpha=160
    relative_mult = 1.0
    letter_dissolve[:, :, 3] = (letter_dissolve[:, :, 3] * relative_mult).astype(np.uint8)
    print(f"Dissolve sprite alpha: max={letter_dissolve[:,:,3].max()}")
    
    # Dissolve pastes onto fresh background
    letter_img2 = Image.fromarray(letter_dissolve)
    canvas_dissolve.paste(letter_img2, (75, 75), letter_img2)
    
    # Dissolve returns RGB only  
    dissolve_result = np.array(canvas_dissolve)[:, :, :3]
    
    # Check the actual pixel values
    letter_region2 = dissolve_result[85:125, 85:125]
    letter_pixels2 = letter_region2[letter_region2 > 50]
    if len(letter_pixels2) > 0:
        print(f"Dissolve result: letter pixels = {letter_pixels2[0]} (on background=50)")
    
    print("\n=== COMPARISON ===")
    if len(letter_pixels) > 0 and len(letter_pixels2) > 0:
        print(f"Motion frame 18: letter value = {letter_pixels[0]}")
        print(f"Dissolve frame 19: letter value = {letter_pixels2[0]}")
        if letter_pixels[0] == letter_pixels2[0]:
            print("✅ No blink - values match!")
        else:
            print(f"❌ BLINK! Values differ by {abs(int(letter_pixels[0]) - int(letter_pixels2[0]))}")
    
    # Save images for visual inspection
    cv2.imwrite('outputs/motion_frame18.png', cv2.cvtColor(motion_result, cv2.COLOR_RGB2BGR))
    cv2.imwrite('outputs/dissolve_frame19.png', cv2.cvtColor(dissolve_result, cv2.COLOR_RGB2BGR))
    print("\nSaved: outputs/motion_frame18.png and outputs/dissolve_frame19.png")

simulate_handoff()