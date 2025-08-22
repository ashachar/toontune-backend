#!/usr/bin/env python3
"""
Fix to ensure NO static text is left behind when letters dissolve.
Each letter should ONLY exist as the dissolving sprite moving upward.
"""

import os

# Read word_dissolve.py
with open('utils/animations/word_dissolve.py', 'r') as f:
    content = f.read()

# Find the section where we decide whether to render base text
# We need to be MORE aggressive - don't render ANY base text once dissolve starts
old_section = """        # (3) Compose the base frozen text with PERMANENT + FRAME holes
        # CRITICAL: Skip base text rendering entirely if all letters are dissolving
        if len(not_started) == 0:
            self._dbg(f"frame={frame_idx}: not_started={len(not_started)}, SKIPPING base text render")
            # All letters have started dissolving - skip base text completely
            pass  # Don't modify result at all
        else:"""

new_section = """        # (3) Compose the base frozen text with PERMANENT + FRAME holes
        # CRITICAL: NEVER render base text during dissolve phase
        # Each letter should ONLY appear as its dissolving sprite, not static
        if len(dissolving) > 0 or len(completed) > 0:
            self._dbg(f"frame={frame_idx}: Letters dissolving/completed, NO base text")
            # Some letters are dissolving - don't render ANY base text
            # Only the floating sprites should be visible
            pass  # Don't modify result at all
        else:"""

if old_section in content:
    content = content.replace(old_section, new_section)
    with open('utils/animations/word_dissolve.py', 'w') as f:
        f.write(content)
    print("✓ Applied fix: NO base text when ANY letter is dissolving")
    print("  Letters will ONLY appear as floating sprites during dissolve")
else:
    print("Could not find exact pattern, trying alternative...")
    
    # Alternative approach
    alt_old = """        if len(not_started) == 0:"""
    alt_new = """        if len(dissolving) > 0 or len(completed) > 0:"""
    
    if alt_old in content:
        content = content.replace(alt_old, alt_new)
        with open('utils/animations/word_dissolve.py', 'w') as f:
            f.write(content)
        print("✓ Applied alternative fix")
    else:
        print("Could not apply fix automatically")