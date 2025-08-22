#!/usr/bin/env python3
"""
Fix for letters remaining visible after dissolve completion.

The issue: Completed letters are still partially visible because:
1. The dead_mask might not fully cover them
2. The hole masks might not be fully opaque

Solution: Ensure completed letters are FULLY removed from base text rendering.
"""

import os

# Read the current word_dissolve.py
with open('utils/animations/word_dissolve.py', 'r') as f:
    lines = f.readlines()

# Find and modify the section where completed letters are handled
# We need to ensure the dead_mask fully covers completed letters

modified = False
for i, line in enumerate(lines):
    # Find where we handle completed letters in the dead_mask growth
    if "# (1) Grow the persistent kill mask exactly when a letter STARTS dissolving" in line:
        # Find the end of this section (around line 555)
        j = i
        while j < len(lines) and not lines[j].strip().startswith("# (2)"):
            j += 1
        
        # Insert code to also ensure completed letters are fully masked
        insert_point = j - 1
        new_code = """
        # Also ensure completed letters are FULLY masked (opacity = 1.0)
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
                sub[sprite_alpha > 0.01] = 1.0
"""
        lines.insert(insert_point, new_code)
        modified = True
        print(f"Added completed letter masking at line {insert_point}")
        break

if not modified:
    # Alternative approach - modify the section where holes are combined
    for i, line in enumerate(lines):
        if "# (3) Compose the base frozen text with PERMANENT + FRAME holes" in line:
            # Find where base_a is modified
            j = i
            while j < len(lines) and "if bg_vis is not None:" not in lines[j]:
                j += 1
            
            # Insert before bg_vis check
            new_code = """        # Ensure completed letters are fully removed
        for i in completed:
            S = self.letter_sprites[i] 
            if S:
                x0, y0, x1, y1 = S["orig_bbox"] if "orig_bbox" in S else S["bbox"]
                # Expand bounds slightly to catch any edge pixels
                x0, y0 = max(0, x0 - 2), max(0, y0 - 2)
                x1, y1 = min(W, x1 + 2), min(H, y1 + 2)
                base_a[y0:y1, x0:x1] = 0.0  # Force to fully transparent
        
"""
            lines.insert(j, new_code)
            modified = True
            print(f"Added alternative completed letter removal at line {j}")
            break

if modified:
    # Write the updated file
    with open('utils/animations/word_dissolve.py', 'w') as f:
        f.writelines(lines)
    print("\nApplied fix for completed letters remaining visible")
    print("Completed letters will now be fully removed from the base text layer")
else:
    print("Could not find the target section to modify")
    print("Applying manual fix...")
    
    # Manual approach - directly edit the specific section
    content = ''.join(lines)
    
    # Find and replace the dead_mask section to include completed letters
    old_section = """                        self._dbg(f"kill-mask grow: idx={i} letter='{self.word[i]}' radius={radius}px bbox=({x0},{y0},{x1},{y1})")

        # (2) Build a frame-local hole"""
    
    new_section = """                        self._dbg(f"kill-mask grow: idx={i} letter='{self.word[i]}' radius={radius}px bbox=({x0},{y0},{x1},{y1})")
        
        # Ensure completed letters are FULLY removed
        for i in completed:
            S = self.letter_sprites[i]
            if S:
                x0, y0, x1, y1 = S["bbox"]
                # Force dead mask to 1.0 for completed letters
                self._dead_mask[y0:y1, x0:x1] = 1.0

        # (2) Build a frame-local hole"""
    
    if old_section in content:
        content = content.replace(old_section, new_section)
        with open('utils/animations/word_dissolve.py', 'w') as f:
            f.write(content)
        print("\nApplied manual fix for completed letters")
    else:
        print("Manual fix pattern not found")