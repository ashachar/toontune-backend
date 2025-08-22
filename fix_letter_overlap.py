#!/usr/bin/env python3
"""
Fix for adjacent letters disappearing when a letter starts dissolving.

The issue: Rectangular bounding box expansion overlaps with adjacent letters.
Solution: Use the actual sprite alpha mask instead of rectangular regions.
"""

import os

# Read the current word_dissolve.py
with open('utils/animations/word_dissolve.py', 'r') as f:
    content = f.read()

# Find and replace the base letter mask section
old_section = """        # Create mask for letters that should be visible in base
        # Only show letters in not_started state
        base_letter_mask = np.ones((H, W), dtype=np.float32)
        for i in dissolving + completed:
            S = self.letter_sprites[i]
            if S:
                x0, y0, x1, y1 = S["bbox"]
                # Expand to include outlines
                expand = self.outline_width + 3
                x0 = max(0, x0 - expand)
                y0 = max(0, y0 - expand)
                x1 = min(W, x1 + expand)
                y1 = min(H, y1 + expand)
                base_letter_mask[y0:y1, x0:x1] = 0.0"""

new_section = """        # Create mask for letters that should be visible in base
        # Only show letters in not_started state
        base_letter_mask = np.ones((H, W), dtype=np.float32)
        for i in dissolving + completed:
            S = self.letter_sprites[i]
            if S:
                x0, y0, x1, y1 = S["bbox"]
                # Use the actual sprite alpha to avoid overlapping adjacent letters
                sprite_alpha = S["sprite"][:, :, 3].astype(np.float32) / 255.0
                
                # Apply slight dilation to cover outlines
                from PIL import Image, ImageFilter
                alpha_img = Image.fromarray((sprite_alpha * 255).astype(np.uint8))
                # Small dilation to cover outlines without overlapping neighbors
                dilated = alpha_img.filter(ImageFilter.MaxFilter(size=5))
                sprite_alpha_dilated = np.array(dilated).astype(np.float32) / 255.0
                
                # Subtract this sprite from the base mask
                mask_region = base_letter_mask[y0:y1, x0:x1]
                np.minimum(mask_region, 1.0 - sprite_alpha_dilated, out=mask_region)"""

if old_section in content:
    content = content.replace(old_section, new_section)
    with open('utils/animations/word_dissolve.py', 'w') as f:
        f.write(content)
    print("✓ Applied fix for letter overlap")
    print("  Now using actual sprite alpha masks instead of rectangular bounds")
    print("  Adjacent letters won't be affected when a letter starts dissolving")
else:
    print("Pattern not found. Trying simpler fix...")
    
    # Simpler approach - just reduce the expansion
    old_expand = """                expand = self.outline_width + 3"""
    new_expand = """                expand = 1  # Minimal expansion to avoid overlap"""
    
    if old_expand in content:
        content = content.replace(old_expand, new_expand)
        with open('utils/animations/word_dissolve.py', 'w') as f:
            f.write(content)
        print("✓ Applied simpler fix - reduced expansion to avoid overlap")
    else:
        print("Could not apply fix")