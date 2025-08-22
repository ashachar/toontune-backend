#!/usr/bin/env python3
"""
Fix for base text remaining visible during dissolve.

The issue: Base text with 50% alpha is still being rendered during dissolve.
Solution: Don't render ANY base text when letters are dissolving.
"""

import os

# Read the current word_dissolve.py
with open('utils/animations/word_dissolve.py', 'r') as f:
    content = f.read()

# Find the section where base text is rendered
old_section = """        # Apply the mask to remove dissolving/completed letters from base
        base_a = base_a * base_letter_mask
        
        if bg_vis is not None:
            base_a = base_a * bg_vis

        base_f = result.astype(np.float32)
        out = base_f * (1.0 - base_a[..., None]) + base_rgb * base_a[..., None]
        result = np.clip(out, 0, 255).astype(np.uint8)"""

new_section = """        # Apply the mask to remove dissolving/completed letters from base
        base_a = base_a * base_letter_mask
        
        # CRITICAL: If ANY letters are dissolving, don't render base text at all
        # Only render base text if there are letters that haven't started dissolving yet
        if len(not_started) == 0:
            # All letters have started dissolving - don't render base text
            base_a = np.zeros_like(base_a)
        
        if bg_vis is not None:
            base_a = base_a * bg_vis

        base_f = result.astype(np.float32)
        out = base_f * (1.0 - base_a[..., None]) + base_rgb * base_a[..., None]
        result = np.clip(out, 0, 255).astype(np.uint8)"""

if old_section in content:
    content = content.replace(old_section, new_section)
    with open('utils/animations/word_dissolve.py', 'w') as f:
        f.write(content)
    print("✓ Applied fix for base text during dissolve")
    print("  Base text is now completely hidden when all letters are dissolving")
else:
    print("Pattern not found. Trying alternative approach...")
    
    # Alternative: Find and modify differently
    lines = content.split('\n')
    for i, line in enumerate(lines):
        if '# Apply the mask to remove dissolving/completed letters from base' in line:
            # Insert check after this line
            insert_lines = [
                "        ",
                "        # CRITICAL: If no letters are waiting to start, hide all base text",
                "        if len(not_started) == 0:",
                "            base_a = np.zeros_like(base_a)",
            ]
            
            for j, new_line in enumerate(insert_lines):
                lines.insert(i + 2 + j, new_line)
            
            content = '\n'.join(lines)
            with open('utils/animations/word_dissolve.py', 'w') as f:
                f.write(content)
            print("✓ Applied alternative fix")
            break