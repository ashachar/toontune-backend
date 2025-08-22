#!/usr/bin/env python3
"""
Fix to make each letter disappear from base text only when its specific dissolve starts.
Letters that haven't started dissolving yet should remain visible in base.
"""

import os

# Read word_dissolve.py
with open('utils/animations/word_dissolve.py', 'r') as f:
    content = f.read()

# Find and replace the base text rendering logic
old_section = """        # (3) Compose the base frozen text with PERMANENT + FRAME holes
        # CRITICAL: NEVER render base text during dissolve phase
        # Each letter should ONLY appear as its dissolving sprite, not static
        if len(dissolving) > 0 or len(completed) > 0:
            self._dbg(f"frame={frame_idx}: Letters dissolving/completed, NO base text")
            # Some letters are dissolving - don't render ANY base text
            # Only the floating sprites should be visible
            pass  # Don't modify result at all
        else:
            # Some letters haven't started yet - render only those
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
                    # Use the actual sprite alpha to avoid overlapping adjacent letters
                    sprite_alpha = S["sprite"][:, :, 3].astype(np.float32) / 255.0
                    
                    # Apply slight dilation to cover outlines
                    alpha_img = Image.fromarray((sprite_alpha * 255).astype(np.uint8))
                    # Small dilation to cover outlines without overlapping neighbors
                    dilated = alpha_img.filter(ImageFilter.MaxFilter(size=5))
                    sprite_alpha_dilated = np.array(dilated).astype(np.float32) / 255.0
                    
                    # Subtract this sprite from the base mask
                    mask_region = base_letter_mask[y0:y1, x0:x1]
                    np.minimum(mask_region, 1.0 - sprite_alpha_dilated, out=mask_region)
            
            # Apply the mask to remove dissolving/completed letters from base
            base_a = base_a * base_letter_mask
            
            if bg_vis is not None:
                base_a = base_a * bg_vis

            base_f = result.astype(np.float32)
            out = base_f * (1.0 - base_a[..., None]) + base_rgb * base_a[..., None]
            result = np.clip(out, 0, 255).astype(np.uint8)"""

new_section = """        # (3) Compose the base frozen text with per-letter masking
        # Each letter disappears from base ONLY when its specific dissolve starts
        if len(not_started) > 0:
            # Some letters haven't started - render only those
            base_rgba = self.frozen_text_rgba
            base_rgb = base_rgba[:, :, :3].astype(np.float32)
            base_a = base_rgba[:, :, 3].astype(np.float32) / 255.0

            # Create mask to hide letters that are dissolving or completed
            # Keep letters that haven't started yet
            base_letter_mask = np.ones((H, W), dtype=np.float32)
            
            for i in dissolving + completed:
                S = self.letter_sprites[i]
                if S:
                    x0, y0, x1, y1 = S["bbox"]
                    # Use the actual sprite alpha to avoid overlapping adjacent letters
                    sprite_alpha = S["sprite"][:, :, 3].astype(np.float32) / 255.0
                    
                    # Apply slight dilation to cover outlines
                    alpha_img = Image.fromarray((sprite_alpha * 255).astype(np.uint8))
                    # Small dilation to cover outlines without overlapping neighbors
                    dilated = alpha_img.filter(ImageFilter.MaxFilter(size=5))
                    sprite_alpha_dilated = np.array(dilated).astype(np.float32) / 255.0
                    
                    # Subtract this sprite from the base mask
                    mask_region = base_letter_mask[y0:y1, x0:x1]
                    np.minimum(mask_region, 1.0 - sprite_alpha_dilated, out=mask_region)
            
            # Apply the mask to remove dissolving/completed letters from base
            base_a = base_a * base_letter_mask
            
            if bg_vis is not None:
                base_a = base_a * bg_vis

            base_f = result.astype(np.float32)
            out = base_f * (1.0 - base_a[..., None]) + base_rgb * base_a[..., None]
            result = np.clip(out, 0, 255).astype(np.uint8)
            
            self._dbg(f"frame={frame_idx}: Rendering {len(not_started)} letters in base, hiding {len(dissolving)} dissolving")
        else:
            # All letters have started dissolving - no base text
            self._dbg(f"frame={frame_idx}: All letters dissolving/completed, NO base text")"""

if old_section in content:
    content = content.replace(old_section, new_section)
    with open('utils/animations/word_dissolve.py', 'w') as f:
        f.write(content)
    print("✓ Applied fix: Per-letter timing")
    print("  Each letter remains in base until its specific dissolve starts")
    print("  Only dissolving letters are hidden from base and shown as sprites")
else:
    print("Could not find exact pattern")
    print("This might mean the code has already been modified")