#!/usr/bin/env python3
"""Analyze which letters are visible in each frame."""

# Dissolve parameters
fps = 60  # Note: FPS is 60 in the actual test
dissolve_duration = 1.5
stable_duration = 0.1
dissolve_stagger = 0.1
dissolve_window = 0.5  # Each letter dissolves for 0.5s
fade_duration = 0.05  # New fade period

total_frames = int(dissolve_duration * fps)  # 90 frames at 60fps

text = "HELLOWORLD"
dissolve_order = [0, 1, 2, 3, 4, 6, 7, 8, 9, 10]  # Excludes index 5 (space)

print(f"Total frames: {total_frames}")
print(f"FPS: {fps}")
print("=" * 80)

# Focus on frames around the jump
test_frames = list(range(30, 41))

for frame in test_frames:
    t = frame / (total_frames - 1)
    time_s = frame / fps
    
    visible = []
    fading = []
    dissolving = []
    gone = []
    
    for idx in dissolve_order:
        letter = text[idx] if idx < 5 else text[idx-1]  # Adjust for space
        
        letter_order_idx = dissolve_order.index(idx)
        letter_start = (stable_duration + letter_order_idx * dissolve_stagger) / dissolve_duration
        letter_end = letter_start + dissolve_window / dissolve_duration
        letter_fade_end = letter_end + fade_duration / dissolve_duration
        
        if t < letter_start:
            visible.append(letter)
        elif t > letter_fade_end:
            gone.append(letter)
        elif t > letter_end:
            # In fade period
            fade_t = (t - letter_end) / (fade_duration / dissolve_duration)
            alpha = 0.02 * (1.0 - fade_t)
            fading.append(f"{letter}({alpha:.3f})")
        else:
            # Dissolving
            letter_t = (t - letter_start) / (letter_end - letter_start)
            alpha = 0.5 * (1.0 - letter_t)  # stable_alpha = 0.5
            dissolving.append(f"{letter}({alpha:.2f})")
    
    print(f"Frame {frame:3d} (t={t:.4f}, {time_s:.3f}s):")
    print(f"  Visible: {''.join(visible):10s} | Dissolving: {', '.join(dissolving):30s} | Fading: {', '.join(fading):15s} | Gone: {''.join(gone)}")
    
    # Highlight the jump frame
    if frame == 34:
        print("  ^^^ BEFORE JUMP ^^^")
    elif frame == 35:
        print("  vvv AFTER JUMP vvv")