#!/usr/bin/env python3
"""Verify the settling phase alpha bug."""

# Frame 18 is in settling phase
frame_number = 18
shrink_frames = 15
settle_frames = 4
final_alpha = 0.5

# Settling phase calculation (from line 275-281)
t = (frame_number - shrink_frames) / max(1, settle_frames)
t_smooth = t * t * (3 - 2 * t)

print(f"Settling phase:")
print(f"t = ({frame_number} - {shrink_frames}) / {settle_frames} = {t:.3f}")
print(f"t_smooth = {t_smooth:.3f}")

# Current WRONG calculation (line 281)
alpha_wrong = 1.0 + (final_alpha - 1.0) * t_smooth
print(f"\nCurrent (WRONG) calculation:")
print(f"alpha = 1.0 + ({final_alpha} - 1.0) * {t_smooth:.3f}")
print(f"alpha = 1.0 + (-0.5) * {t_smooth:.3f}")
print(f"alpha = {alpha_wrong:.3f}")

# What it SHOULD be
alpha_correct = final_alpha
print(f"\nCorrect calculation:")
print(f"alpha = {alpha_correct:.3f} (should stay at final_alpha)")

print(f"\n❌ BUG FOUND! Settling phase continues to change alpha!")
print(f"At frame 18: alpha={alpha_wrong:.3f} instead of {alpha_correct:.3f}")
print(f"This explains why debug shows 0.630 instead of 0.500!")