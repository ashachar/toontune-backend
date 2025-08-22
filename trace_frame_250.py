#!/usr/bin/env python3
"""Trace what happens at frame 250 in the test."""

# Test parameters
phase1_frames = 20
phase2_frames = 15  
phase3_frames = 25
fps = 60

# WordDissolve parameters
stable_duration = 0.1
dissolve_duration = 1.0
dissolve_stagger = 0.25
num_letters = 11  # "HELLO WORLD"

# Calculate
stable_frames = int(stable_duration * fps)  # 6
dissolve_frames = int(dissolve_duration * fps)  # 60
stagger_frames = int(dissolve_stagger * fps)  # 15

wd_duration = stable_frames + 10 * stagger_frames + dissolve_frames  # 216
total_frames = phase1_frames + phase2_frames + phase3_frames + wd_duration + 30  # 306

print("Frame timeline:")
print(f"  Frames 0-{phase1_frames-1}: Phase 1 (shrink)")
print(f"  Frames {phase1_frames}-{phase1_frames+phase2_frames-1}: Phase 2 (move behind)")
print(f"  Frames {phase1_frames+phase2_frames}-{phase1_frames+phase2_frames+phase3_frames-1}: Phase 3 (stable behind)")

wd_start = phase1_frames + phase2_frames + phase3_frames
print(f"  Frames {wd_start}-{wd_start+wd_duration-1}: WordDissolve")
print(f"  Frames {wd_start+wd_duration}-{total_frames-1}: Extra frames")

print(f"\nFrame 250 analysis:")
print(f"  Global frame: 250")
print(f"  WordDissolve start frame: {wd_start}")

if 250 < wd_start:
    print(f"  Status: TextBehindSegment phase")
elif 250 < wd_start + wd_duration:
    dissolve_frame = 250 - wd_start
    print(f"  Status: WordDissolve frame {dissolve_frame}")
    print(f"  WordDissolve completes at frame {wd_duration}")
    if dissolve_frame >= wd_duration:
        print("  -> All letters should be completed!")
else:
    dissolve_frame = 250 - wd_start
    print(f"  Status: After WordDissolve (frame {dissolve_frame} in dissolve)")
    print(f"  This is WordDissolve frame: {dissolve_frame}")
    print(f"  WordDissolve ended at frame {wd_duration}")
    
    # Check what the test code does here
    print("\nIn test code, this frame goes through:")
    print("  dissolve_frame = 250 - 60 = 190")
    print("  word_dissolver.render_word_frame(frame, 190, mask)")
    print(f"  At dissolve frame 190, all letters completed at frame {wd_duration}=216")
    print("  So frame 190 is in the 'all completed' range")