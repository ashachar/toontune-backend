#!/usr/bin/env python3
"""Debug the wiggle grow/shrink issue"""

import numpy as np
import matplotlib.pyplot as plt

print("="*60)
print("DEBUGGING WIGGLE GROW/SHRINK ISSUE")
print("="*60)

# Analyze the wiggle phase
print("\nCurrent wiggle implementation:")
print("  wiggle_amount = np.sin(t * np.pi * 4) * 0.02")
print("  scale = end_scale * (1.0 + wiggle_amount)")
print("")

# Simulate wiggle phase
end_scale = 1.0
wiggle_frames = 9  # Typical wiggle duration
fps = 60

print("Frame-by-frame analysis of wiggle phase:")
print("-" * 50)
print("Frame | t     | sin(t*π*4) | wiggle | scale  | Change")
print("-" * 50)

scales = []
for i in range(wiggle_frames):
    t = i / max(wiggle_frames - 1, 1)
    sin_val = np.sin(t * np.pi * 4)
    wiggle_amount = sin_val * 0.02
    scale = end_scale * (1.0 + wiggle_amount)
    scales.append(scale)
    
    if i == 0:
        change = "START"
    elif scale > scales[i-1]:
        change = "↑ GROWS!"
    elif scale < scales[i-1]:
        change = "↓ shrinks"
    else:
        change = "= same"
    
    print(f"{i:5d} | {t:.3f} | {sin_val:10.3f} | {wiggle_amount:6.3f} | {scale:.4f} | {change}")

print("\n" + "="*60)
print("THE PROBLEM:")
print("="*60)
print("\n❌ The sine wave goes both positive AND negative")
print("❌ Positive values make scale > end_scale (GROWTH)")
print("❌ Negative values make scale < end_scale (shrink)")
print("❌ This creates alternating grow/shrink behavior")

# Show the visual pattern
print("\nVisual representation of scale changes:")
print("-" * 50)
for i, scale in enumerate(scales):
    bar = "█" * int((scale - 0.98) * 500)
    print(f"Frame {i}: {scale:.4f} |{bar}")

print("\n" + "="*60)
print("THE SOLUTION:")
print("="*60)
print("\n✅ Option 1: Only allow shrinking")
print("  wiggle_amount = -abs(np.sin(t * np.pi * 4)) * 0.02")
print("  This makes wiggle always negative")
print("")
print("✅ Option 2: Use a decay instead of wiggle")
print("  scale = end_scale * (1.0 - t * 0.02)")
print("  This creates gradual shrinking")
print("")
print("✅ Option 3: Clamp to never exceed end_scale")
print("  wiggle_amount = min(0, np.sin(t * np.pi * 4) * 0.02)")
print("  This prevents growth but keeps some variation")

# Test the fixes
print("\n" + "="*60)
print("TESTING FIXES:")
print("="*60)

print("\nOption 1 - Only shrinking wiggle:")
print("-" * 50)
for i in range(5):
    t = i / 4
    wiggle_amount = -abs(np.sin(t * np.pi * 4)) * 0.02
    scale = end_scale * (1.0 + wiggle_amount)
    print(f"Frame {i}: scale = {scale:.4f} (always ≤ {end_scale})")

print("\nOption 2 - Gradual decay:")
print("-" * 50)
for i in range(5):
    t = i / 4
    scale = end_scale * (1.0 - t * 0.02)
    print(f"Frame {i}: scale = {scale:.4f} (monotonic decrease)")

print("\nOption 3 - Clamped wiggle:")
print("-" * 50)
for i in range(5):
    t = i / 4
    wiggle_amount = min(0, np.sin(t * np.pi * 4) * 0.02)
    scale = end_scale * (1.0 + wiggle_amount)
    print(f"Frame {i}: scale = {scale:.4f} (never > {end_scale})")

print("\n📊 Recommendation: Option 1 or 2")
print("   Option 1 keeps some movement but always shrinks")
print("   Option 2 is simplest - just gradual shrinking")