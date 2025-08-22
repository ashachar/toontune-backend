#!/usr/bin/env python3
"""
Add transparency effect during shrink phase - text fades to 0.5 alpha as it shrinks.

This creates a visual cue that the text is moving to the back of the image.
"""

import os

# Read the current text_behind_segment.py
with open('utils/animations/text_behind_segment.py', 'r') as f:
    content = f.read()

# Find where text_alpha is computed and add transparency based on phase
old_section = """        text_layer_np = np.array(text_layer)  # (h, w, 4)
        text_alpha = (text_layer_np[:, :, 3].astype(np.float32) / 255.0)[..., None]  # (h, w, 1)
        text_rgb = text_layer_np[:, :, :3].astype(np.float32)  # (h, w, 3)"""

new_section = """        text_layer_np = np.array(text_layer)  # (h, w, 4)
        text_alpha = (text_layer_np[:, :, 3].astype(np.float32) / 255.0)[..., None]  # (h, w, 1)
        text_rgb = text_layer_np[:, :, :3].astype(np.float32)  # (h, w, 3)
        
        # Apply transparency during shrink phase to show text moving to back
        if phase == "foreground" and frame_idx > 0:
            # During shrink (phase 1), fade from 1.0 to 0.5 alpha
            phase_progress = frame_idx / self.phase1_end if self.phase1_end > 0 else 1
            target_alpha = 1.0 - (0.5 * phase_progress)  # Goes from 1.0 to 0.5
            text_alpha = text_alpha * target_alpha
        elif phase == "transition":
            # During transition (phase 2), maintain 0.5 alpha
            text_alpha = text_alpha * 0.5
        elif phase == "background":
            # During stable behind (phase 3), keep at 0.5 alpha
            # The final fade out will happen during dissolve
            text_alpha = text_alpha * 0.5"""

if old_section in content:
    content = content.replace(old_section, new_section)
    with open('utils/animations/text_behind_segment.py', 'w') as f:
        f.write(content)
    print("✓ Applied transparency effect during shrink phase")
    print("  Text now fades from 100% to 50% opacity while shrinking")
    print("  This creates a visual depth cue that text is moving behind")
else:
    print("Pattern not found exactly. Trying alternative approach...")
    
    # Alternative: Find the section differently
    lines = content.split('\n')
    for i, line in enumerate(lines):
        if 'text_alpha = (text_layer_np[:, :, 3]' in line:
            # Found the line, insert after it
            insert_lines = [
                "        ",
                "        # Apply transparency during shrink phase to show text moving to back",
                "        if phase == \"foreground\" and frame_idx > 0:",
                "            # During shrink (phase 1), fade from 1.0 to 0.5 alpha",
                "            phase_progress = frame_idx / self.phase1_end if self.phase1_end > 0 else 1",
                "            target_alpha = 1.0 - (0.5 * phase_progress)  # Goes from 1.0 to 0.5",
                "            text_alpha = text_alpha * target_alpha",
                "        elif phase == \"transition\":",
                "            # During transition (phase 2), maintain 0.5 alpha",
                "            text_alpha = text_alpha * 0.5",
                "        elif phase == \"background\":",
                "            # During stable behind (phase 3), keep at 0.5 alpha",
                "            # The final fade out will happen during dissolve",
                "            text_alpha = text_alpha * 0.5",
            ]
            
            for j, new_line in enumerate(insert_lines):
                lines.insert(i + 2 + j, new_line)
            
            content = '\n'.join(lines)
            with open('utils/animations/text_behind_segment.py', 'w') as f:
                f.write(content)
            print("✓ Applied transparency effect (alternative method)")
            break