#!/usr/bin/env python3
"""Calculate what alpha motion should have at frame 18."""

# From debug output: Motion: 0.75s (19 frames)
total_frames = 19
shrink_frames = int(19 * 0.8)  # 80% of duration
frame_number = 18  # Last frame

print(f"Total frames: {total_frames}")
print(f"Shrink frames: {shrink_frames}")
print(f"Frame number: {frame_number}")

if frame_number < shrink_frames:
    # Still shrinking
    t = frame_number / max(1, shrink_frames)
    t_smooth = t * t * (3 - 2 * t)  # smoothstep
    
    print(f"\nShrinking phase:")
    print(f"t = {frame_number}/{shrink_frames} = {t:.3f}")
    print(f"t_smooth = {t_smooth:.3f}")
    
    # is_behind = t > 0.0, so it's True
    is_behind = t > 0.0
    print(f"is_behind = {is_behind}")
    
    # Alpha calculation when behind
    final_alpha = 0.5  # Default from the code
    alpha = 1.0 + (final_alpha - 1.0) * t_smooth
    print(f"alpha = 1.0 + ({final_alpha} - 1.0) * {t_smooth:.3f} = {alpha:.3f}")
else:
    # Settling phase
    frame_in_settle = frame_number - shrink_frames
    settle_frames = total_frames - shrink_frames
    t_settle = frame_in_settle / max(1, settle_frames)
    
    print(f"\nSettling phase:")
    print(f"Frame in settle: {frame_in_settle}")
    print(f"Settle frames: {settle_frames}")
    print(f"t_settle = {t_settle:.3f}")
    
    # In settling, alpha stays at final_alpha
    final_alpha = 0.5
    alpha = final_alpha
    print(f"alpha = {alpha:.3f}")

print(f"\n✅ Motion frame 18 should have alpha = {alpha:.3f}")
print(f"But debug shows alpha = 0.630")
print("\nDiscrepancy suggests final_opacity might not be 0.5 or there's additional calculation")