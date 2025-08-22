#!/usr/bin/env python3
"""
Fix for edge artifacts during dissolve animation.

The artifacts are caused by:
1. Glow effect extending beyond the original letter boundaries
2. Scaling operations creating edge pixels
3. Hole masks not fully covering the extended boundaries

Solution: Increase hole mask coverage to account for glow and scale effects.
"""

import os

# Read the current word_dissolve.py
with open('utils/animations/word_dissolve.py', 'r') as f:
    content = f.read()

# Find and replace the hole radius calculation (line ~568)
# Current: hole_radius = int(round(max(dims) * 0.10))
# New: More aggressive radius that accounts for glow + scale

old_hole_radius = """            if dims:
                hole_radius = int(round(max(dims) * 0.10))  # softer/lighter than before"""

new_hole_radius = """            if dims:
                # Increase hole radius to fully cover glow + scale effects
                # Account for: scale growth (max_scale - 1.0), glow blur (up to 16% of size)
                hole_radius = int(round(max(dims) * 0.25))  # Increased from 0.10 to prevent edge artifacts"""

content = content.replace(old_hole_radius, new_hole_radius)

# Also increase the kill mask radius calculation (line ~545)
# This ensures the persistent mask fully covers dissolving letters

old_kill_radius = """                        # radius must cover: outline + glow + scale growth
                        sh, sw = S["sprite"].shape[:2]
                        grow = int(round(max(sh, sw) * (max(self.max_scale, 1.0) - 1.0) * 0.6))
                        glow = int(round(max(sh, sw) * 0.12))
                        radius = max(self.outline_width + 2, glow, grow, 4)"""

new_kill_radius = """                        # radius must cover: outline + glow + scale growth
                        sh, sw = S["sprite"].shape[:2]
                        # Increase growth factor to fully cover scaled sprites
                        grow = int(round(max(sh, sw) * (max(self.max_scale, 1.0) - 1.0) * 0.8))  # Increased from 0.6
                        # Increase glow radius to match actual glow effect
                        glow = int(round(max(sh, sw) * 0.20))  # Increased from 0.12
                        radius = max(self.outline_width + 4, glow, grow, 6)  # Increased minimums"""

content = content.replace(old_kill_radius, new_kill_radius)

# Write the updated file
with open('utils/animations/word_dissolve.py', 'w') as f:
    f.write(content)

print("Applied edge artifact fixes:")
print("1. Increased hole_radius from 0.10 to 0.25 to fully cover glow/scale effects")
print("2. Increased kill mask growth factor from 0.6 to 0.8")
print("3. Increased glow radius factor from 0.12 to 0.20")
print("4. Increased minimum radius values for better coverage")
print("\nThese changes ensure that hole masks fully cover the extended boundaries")
print("created by glow and scale effects, eliminating edge artifacts.")