#!/usr/bin/env python3
"""Add debug print to see what's happening."""

import os

# Read word_dissolve.py
with open('utils/animations/word_dissolve.py', 'r') as f:
    lines = f.readlines()

# Find the line where we check not_started
for i, line in enumerate(lines):
    if 'if len(not_started) == 0:' in line and 'CRITICAL' in lines[i-1]:
        # Insert debug print
        indent = '        '
        debug_line = f'{indent}self._dbg(f"frame={{frame_idx}}: not_started={{len(not_started)}}, SKIPPING base text render")\n'
        lines.insert(i+1, debug_line)
        break

# Write back
with open('utils/animations/word_dissolve.py', 'w') as f:
    f.writelines(lines)

print("Added debug print")