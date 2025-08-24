#!/usr/bin/env python3
"""Debug the exact timing of dissolve for frames 34-35."""

# Dissolve parameters from test script
fps = 30
dissolve_duration = 1.5  # seconds
stable_duration = 0.1    # seconds
dissolve_stagger = 0.1   # seconds per letter
total_frames = int(dissolve_duration * fps)  # 45 frames

# Text without spaces
text = "HELLOWORLD"  # 10 letters
dissolve_order = list(range(10))  # [0,1,2,3,4,5,6,7,8,9]

print("Letter dissolve timing analysis:")
print("=" * 60)

# Check frames 34 and 35
for frame in [34, 35]:
    t = frame / (total_frames - 1)  # 0 to 1
    print(f"\nFrame {frame}: t={t:.4f} (time={frame/fps:.3f}s)")
    
    visible_letters = []
    dissolving_letters = []
    dissolved_letters = []
    
    for idx in dissolve_order:
        letter_order_idx = dissolve_order.index(idx)
        letter_start = (stable_duration + letter_order_idx * dissolve_stagger) / dissolve_duration
        letter_end = letter_start + 0.5 / dissolve_duration  # dissolve_duration=0.5s
        
        if t < letter_start:
            visible_letters.append(text[idx])
        elif t > letter_end:
            dissolved_letters.append(text[idx])
        else:
            letter_t = (t - letter_start) / (letter_end - letter_start)
            dissolving_letters.append((text[idx], letter_t))
    
    print(f"  Stable: {''.join(visible_letters)}")
    print(f"  Dissolving: {dissolving_letters}")
    print(f"  Dissolved: {''.join(dissolved_letters)}")

# Calculate exact transition points
print("\n" + "=" * 60)
print("Letter transition times:")
for i, letter in enumerate(text):
    letter_start = (stable_duration + i * dissolve_stagger) / dissolve_duration
    letter_end = letter_start + 0.5 / dissolve_duration
    start_frame = letter_start * (total_frames - 1)
    end_frame = letter_end * (total_frames - 1)
    print(f"  {letter}: frames {start_frame:.1f}-{end_frame:.1f} (t={letter_start:.3f}-{letter_end:.3f})")