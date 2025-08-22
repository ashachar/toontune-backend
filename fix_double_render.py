#!/usr/bin/env python3
"""
Fix for letters appearing twice during dissolve (static base + moving sprite).

The issue: Base text still shows dissolving letters, creating ugly double rendering.
Solution: Completely exclude dissolving letters from base text rendering.
"""

import os

# Read the current word_dissolve.py
with open('utils/animations/word_dissolve.py', 'r') as f:
    content = f.read()

# Find the section where base text is composed
old_section = """        # (3) Compose the base frozen text with PERMANENT + FRAME holes
        base_rgba = self.frozen_text_rgba
        base_rgb = base_rgba[:, :, :3].astype(np.float32)
        base_a = base_rgba[:, :, 3].astype(np.float32) / 255.0

        if self._dead_mask is not None:
            base_a = base_a * (1.0 - np.clip(self._dead_mask, 0.0, 1.0))
        if frame_hole is not None:
            base_a = base_a * (1.0 - frame_hole)
        if bg_vis is not None:
            base_a = base_a * bg_vis

        base_f = result.astype(np.float32)
        out = base_f * (1.0 - base_a[..., None]) + base_rgb * base_a[..., None]
        result = np.clip(out, 0, 255).astype(np.uint8)"""

new_section = """        # (3) Compose the base frozen text with PERMANENT + FRAME holes
        # CRITICAL: Only render letters that haven't started dissolving yet
        base_rgba = self.frozen_text_rgba
        base_rgb = base_rgba[:, :, :3].astype(np.float32)
        base_a = base_rgba[:, :, 3].astype(np.float32) / 255.0

        # Create mask for letters that should be visible in base
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
                base_letter_mask[y0:y1, x0:x1] = 0.0
        
        # Apply the mask to remove dissolving/completed letters from base
        base_a = base_a * base_letter_mask
        
        if bg_vis is not None:
            base_a = base_a * bg_vis

        base_f = result.astype(np.float32)
        out = base_f * (1.0 - base_a[..., None]) + base_rgb * base_a[..., None]
        result = np.clip(out, 0, 255).astype(np.uint8)"""

if old_section in content:
    content = content.replace(old_section, new_section)
    with open('utils/animations/word_dissolve.py', 'w') as f:
        f.write(content)
    print("✓ Applied fix for double rendering")
    print("  Dissolving letters are now completely excluded from base text")
    print("  Only the animated sprite version is shown")
else:
    print("Pattern not found exactly. Trying alternative approach...")
    
    # Try a simpler fix - just make the frame_hole fully opaque
    old_hole = """        if frame_hole is not None:
            base_a = base_a * (1.0 - frame_hole)"""
    
    new_hole = """        if frame_hole is not None:
            # Make hole fully opaque to completely hide dissolving letters
            base_a = base_a * (1.0 - np.clip(frame_hole * 2.0, 0.0, 1.0))"""
    
    if old_hole in content:
        content = content.replace(old_hole, new_hole)
        with open('utils/animations/word_dissolve.py', 'w') as f:
            f.write(content)
        print("✓ Applied alternative fix")
        print("  Made frame holes more aggressive to hide base letters")
    else:
        print("Could not apply fix automatically")