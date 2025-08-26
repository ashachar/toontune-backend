#!/usr/bin/env python3
"""Find where the mask is being cached/frozen."""

import re

# Check the dissolve code for any caching patterns
with open('utils/animations/letter_3d_dissolve.py', 'r') as f:
    content = f.read()

# Look for patterns that might indicate caching
patterns = [
    r'self\._.*mask',  # Private mask variables
    r'self\..*mask =',  # Mask assignments
    r'if.*mask.*is not None',  # Mask reuse checks
    r'hasattr.*mask',  # Checking for existing mask
    r'getattr.*mask',  # Getting cached mask
]

print("Searching for potential mask caching patterns...")
print("="*60)

for pattern in patterns:
    matches = re.finditer(pattern, content)
    for match in matches:
        # Get line number
        line_num = content[:match.start()].count('\n') + 1
        # Get the line content
        lines = content.split('\n')
        line_content = lines[line_num - 1].strip()
        print(f"Line {line_num}: {line_content}")

print("\n" + "="*60)
print("Now checking if mask is passed incorrectly...")

# Check how mask is passed between animations
with open('utils/animations/apply_3d_text_animation.py', 'r') as f:
    apply_content = f.read()
    
# Look for segment_mask usage
mask_lines = []
for i, line in enumerate(apply_content.split('\n'), 1):
    if 'segment_mask' in line or 'extract_foreground_mask' in line:
        mask_lines.append((i, line.strip()))

print("\nIn apply_3d_text_animation.py:")
for line_num, line in mask_lines[-10:]:  # Last 10 occurrences
    print(f"Line {line_num}: {line}")

print("\n" + "="*60)
print("HYPOTHESIS: The segment_mask is extracted once at the beginning")
print("and passed to both animations, but it should be extracted fresh")
print("for each frame during dissolve.")