#!/usr/bin/env python3
"""Debug and demonstrate the depth reduction issue"""

print("="*60)
print("DEPTH REDUCTION ISSUE ANALYSIS")
print("="*60)

# The problem is clear from looking at the code:
# depth_offset is multiplied by scale in _render_3d_text()

print("\nCurrent implementation:")
print("  depth_off = depth_offset * scale")
print("  reduced_depth = depth_off * 0.2")
print("")
print("When scale goes from 2.0 → 1.0 during shrinking,")
print("the depth also shrinks from 2.0x → 1.0x")
print("")

# Show the issue
print("Frame analysis during transition:")
print("-" * 50)
print("Frame | t     | Scale | Depth multiplier | Visual")
print("-" * 50)

fps = 60
shrink_frames = int(0.6 * fps)  # 36 frames

for frame in [18, 20, 22, 24, 26, 28, 30]:
    if frame < shrink_frames:
        t = frame / (shrink_frames - 1)
        scale = 2.0 - t * (2.0 - 1.0)
        
        # Check if behind
        is_behind = t >= 0.6
        
        depth_mult = scale * 0.2
        visual = "█" * int(depth_mult * 10)
        
        status = "BEHIND" if is_behind else "FRONT "
        print(f"{frame:5} | {t:.3f} | {scale:.3f} | {depth_mult:.3f}x         | {visual} {status}")

print("\n" + "="*60)
print("THE PROBLEM:")
print("="*60)
print("\n❌ At frame 22 (t=0.629): Text goes BEHIND but depth is 1.37x")
print("❌ At frame 26 (t=0.743): Still BEHIND but depth drops to 1.25x")
print("❌ The depth keeps shrinking even after going behind!")

print("\n" + "="*60)
print("THE SOLUTION:")
print("="*60)
print("\n✅ When is_behind=True, use CONSTANT depth")
print("✅ Keep depth at the value it had when it went behind")
print("✅ This maintains visual consistency")

print("\nProposed fix:")
print("-" * 50)
print("Frame | t     | Scale | Depth multiplier | Visual")
print("-" * 50)

for frame in [18, 20, 22, 24, 26, 28, 30]:
    if frame < shrink_frames:
        t = frame / (shrink_frames - 1)
        scale = 2.0 - t * (2.0 - 1.0)
        
        # Check if behind
        is_behind = t >= 0.6
        
        if is_behind:
            # Use the scale at transition point (t=0.6)
            transition_scale = 2.0 - 0.6 * (2.0 - 1.0)  # = 1.4
            depth_mult = transition_scale * 0.2
        else:
            depth_mult = scale * 0.2
        
        visual = "█" * int(depth_mult * 10)
        
        status = "BEHIND" if is_behind else "FRONT "
        print(f"{frame:5} | {t:.3f} | {scale:.3f} | {depth_mult:.3f}x         | {visual} {status}")

print("\n✅ Now depth stays constant at 0.280x after going behind!")
print("   This prevents the sudden visual change in depth")