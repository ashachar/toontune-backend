#!/usr/bin/env python3
"""
Fix for white outlines remaining after letters dissolve.

The issue: Letter masks only cover the main text, not the white outlines.
Solution: Expand the dead_mask to include outline pixels when letters complete.
"""

import os

# Read the current word_dissolve.py
with open('utils/animations/word_dissolve.py', 'r') as f:
    content = f.read()

# Find the section where completed letters are added to dead_mask
old_section = """        # Also ensure completed letters are FULLY masked (opacity = 1.0)
        for i in completed:
            S = self.letter_sprites[i]
            if S:
                x0, y0, x1, y1 = S["bbox"]
                # Set to full opacity (1.0) for completed letters - they should be invisible
                sub = self._dead_mask[y0:y1, x0:x1]
                sprite_alpha = S["sprite"][:, :, 3].astype(np.float32) / 255.0
                # Use maximum to preserve any existing mask, but ensure at least sprite coverage
                np.maximum(sub, sprite_alpha, out=sub)
                # Force to 1.0 where sprite has any alpha to ensure complete removal
                sub[sprite_alpha > 0.01] = 1.0"""

new_section = """        # Also ensure completed letters are FULLY masked (opacity = 1.0)
        # CRITICAL: Expand mask to cover outlines and shadows
        for i in completed:
            S = self.letter_sprites[i]
            if S:
                x0, y0, x1, y1 = S["bbox"]
                # Expand bounds to cover outline and shadow
                outline_expand = self.outline_width + 3  # Extra pixels for safety
                shadow_expand = max(self.shadow_offset, 3)
                x0 = max(0, x0 - outline_expand)
                y0 = max(0, y0 - outline_expand)
                x1 = min(W, x1 + outline_expand + shadow_expand)
                y1 = min(H, y1 + outline_expand + shadow_expand)
                
                # Force entire expanded region to be fully masked
                self._dead_mask[y0:y1, x0:x1] = 1.0"""

if old_section in content:
    content = content.replace(old_section, new_section)
    with open('utils/animations/word_dissolve.py', 'w') as f:
        f.write(content)
    print("✓ Applied fix for outline remnants")
    print("  Completed letters now mask an expanded area to include outlines and shadows")
else:
    print("Trying alternative approach...")
    
    # Alternative: modify the early return to ensure no text rendering
    old_return = """        # CRITICAL FIX: If ALL letters are completed, return frame without any text
        if len(completed) == len(self.word):
            self._dbg(f"frame={frame_idx}: All letters completed, returning clean frame")
            return result  # result is frame.copy() from line 511"""
    
    new_return = """        # CRITICAL FIX: If ALL letters are completed, return frame without any text
        if len(completed) == len(self.word):
            self._dbg(f"frame={frame_idx}: All letters completed, returning clean frame")
            # Ensure we're not accidentally compositing any frozen text
            return frame.copy()  # Return the original frame, not result"""
    
    if old_return in content:
        content = content.replace(old_return, new_return)
        with open('utils/animations/word_dissolve.py', 'w') as f:
            f.write(content)
        print("✓ Applied alternative fix")
        print("  Now returns original frame instead of result when all completed")
    else:
        print("Could not find patterns to replace")