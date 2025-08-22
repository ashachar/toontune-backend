#!/usr/bin/env python3
"""Trace what's rendering the text at frame 250."""

# The test video timeline:
# Frames 0-59: TextBehindSegment (phases 1-3)
# Frames 60+: WordDissolve

frame = 250
phase1_frames = 20
phase2_frames = 15  
phase3_frames = 25
total_tbs = phase1_frames + phase2_frames + phase3_frames  # 60

print(f"Frame {frame} analysis:")
print(f"  Total TBS frames: {total_tbs}")

if frame < total_tbs:
    print(f"  -> TextBehindSegment renders this frame")
    print(f"     TBS frame: {frame}")
else:
    print(f"  -> WordDissolve renders this frame")
    print(f"     WD frame: {frame - total_tbs}")
    
print()
print("At frame 250:")
print("  WordDissolve should be rendering frame 190")
print("  At WD frame 190, all letters should be dissolving")
print()

# Check if maybe the issue is the handoff data
print("Checking handoff data...")
print("TextBehindSegment freezes text at end of phase 3 with 50% alpha")
print("This frozen RGBA is passed to WordDissolve")
print()
print("HYPOTHESIS: The 50% alpha frozen text from TBS is being rendered")
print("even when WordDissolve thinks it's hiding the base text.")
print()
print("The issue might be that when not_started is empty,")
print("we zero out base_a, but the frozen_text_rgba already has")
print("the 50% alpha baked in from TBS!")