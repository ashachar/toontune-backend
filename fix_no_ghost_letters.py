#!/usr/bin/env python3
"""
Fix to ensure NO ghost letters remain at original position when dissolving.
The letter should ONLY exist as the upward-floating sprite, not in the base.
"""

import os

# Read word_dissolve.py
with open('utils/animations/word_dissolve.py', 'r') as f:
    content = f.read()

# Find the section where we mask out dissolving letters from base
old_section = """                    # Apply slight dilation to cover outlines
                    alpha_img = Image.fromarray((sprite_alpha * 255).astype(np.uint8))
                    # Small dilation to cover outlines without overlapping neighbors
                    dilated = alpha_img.filter(ImageFilter.MaxFilter(size=5))
                    sprite_alpha_dilated = np.array(dilated).astype(np.float32) / 255.0
                    
                    # Subtract this sprite from the base mask
                    mask_region = base_letter_mask[y0:y1, x0:x1]
                    np.minimum(mask_region, 1.0 - sprite_alpha_dilated, out=mask_region)"""

new_section = """                    # AGGRESSIVE dilation to FULLY remove any trace of the letter
                    # This ensures NO ghost remains at the original position
                    alpha_img = Image.fromarray((sprite_alpha * 255).astype(np.uint8))
                    # Larger dilation to cover ALL traces including anti-aliasing and glow
                    dilated = alpha_img.filter(ImageFilter.MaxFilter(size=11))
                    sprite_alpha_dilated = np.array(dilated).astype(np.float32) / 255.0
                    
                    # Threshold to make it binary - either fully masked or not
                    # This prevents partial transparency creating ghosts
                    sprite_alpha_dilated = (sprite_alpha_dilated > 0.1).astype(np.float32)
                    
                    # Completely remove this sprite area from the base mask
                    mask_region = base_letter_mask[y0:y1, x0:x1]
                    mask_region *= (1.0 - sprite_alpha_dilated)"""

if old_section in content:
    content = content.replace(old_section, new_section)
    with open('utils/animations/word_dissolve.py', 'w') as f:
        f.write(content)
    print("✓ Applied fix: No ghost letters")
    print("  Letters are FULLY removed from base when dissolving")
    print("  Larger dilation (11px) ensures complete removal")
    print("  Binary threshold prevents partial transparency ghosts")
else:
    print("Could not find exact pattern")
    print("Trying alternative approach...")
    
    # Try to find and replace just the dilation size
    alt_old = "dilated = alpha_img.filter(ImageFilter.MaxFilter(size=5))"
    alt_new = "dilated = alpha_img.filter(ImageFilter.MaxFilter(size=11))"
    
    if alt_old in content:
        content = content.replace(alt_old, alt_new)
        
        # Also add thresholding after dilation
        thresh_old = "sprite_alpha_dilated = np.array(dilated).astype(np.float32) / 255.0"
        thresh_new = """sprite_alpha_dilated = np.array(dilated).astype(np.float32) / 255.0
                    # Threshold to make it binary - either fully masked or not
                    sprite_alpha_dilated = (sprite_alpha_dilated > 0.1).astype(np.float32)"""
        
        content = content.replace(thresh_old, thresh_new)
        
        # Change the mask application to multiplication
        mask_old = "np.minimum(mask_region, 1.0 - sprite_alpha_dilated, out=mask_region)"
        mask_new = "mask_region *= (1.0 - sprite_alpha_dilated)"
        
        content = content.replace(mask_old, mask_new)
        
        with open('utils/animations/word_dissolve.py', 'w') as f:
            f.write(content)
        print("✓ Applied alternative fix")
    else:
        print("Could not apply fix automatically")