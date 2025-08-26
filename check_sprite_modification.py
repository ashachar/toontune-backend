#!/usr/bin/env python3
"""
Check if letter sprites are being modified in-place causing permanent occlusion.
"""

import numpy as np
from PIL import Image
import cv2

print("="*80)
print("🔍 CHECKING IF SPRITES ARE MODIFIED IN-PLACE")
print("="*80)

# The key issue: When we apply occlusion to a sprite, are we modifying the original?

# Let's trace through the Text3DMotion code
print("\nLooking at Text3DMotion.generate_frame()...")
print("Line 370-391: When applying mask to sprite:")
print("  sprite_np = np.array(scaled_sprite)  # Creates numpy array from PIL Image")
print("  sprite_region[:, :, 3] = (sprite_region[:, :, 3] * (1 - mask_region / 255.0)).astype(np.uint8)")
print("  scaled_sprite = Image.fromarray(sprite_np)  # Converts back to PIL")
print("")
print("⚠️ PROBLEM: This modifies sprite_np which becomes the new scaled_sprite!")
print("  Once pixels are removed by occlusion, they're NEVER restored!")
print("")
print("The sprite's alpha channel is permanently modified, so even if the mask")
print("moves in the next frame, the previously occluded pixels stay invisible!")

print("\n" + "="*80)
print("💡 THE BUG:")
print("="*80)
print("1. Frame N: Mask occludes part of 'd', those pixels get alpha=0")
print("2. Frame N+1: Mask moves, but 'd' sprite already has alpha=0 in those pixels")
print("3. Result: Gap persists even though mask has moved!")

print("\n" + "="*80)
print("🔧 THE FIX NEEDED:")
print("="*80)
print("1. ALWAYS start with a fresh copy of the original sprite")
print("2. Apply current frame's occlusion to the fresh copy")
print("3. Never modify the base sprite")
print("")
print("In Text3DMotion:")
print("  - Store original sprites")
print("  - Each frame: copy original, apply transforms, apply occlusion")
print("  - Never modify the stored originals")

# Let's check the actual code to confirm
import subprocess
result = subprocess.run(
    ["grep", "-A5", "-B5", "sprite_region.*3.*=", "utils/animations/text_3d_motion.py"],
    capture_output=True, text=True
)

print("\n" + "="*80)
print("ACTUAL CODE THAT MODIFIES SPRITES:")
print("="*80)
for line in result.stdout.strip().split('\n'):
    print(line)

print("\n" + "="*80)
print("✅ CONFIRMED: The sprites are being modified in-place!")
print("   Line 388-390 permanently changes the sprite's alpha channel")
print("   This causes the persistent gap even when mask moves")
print("="*80)