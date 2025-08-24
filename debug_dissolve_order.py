#!/usr/bin/env python3
"""Debug the dissolve order and timing."""

text = "HELLO WORLD"
stable_duration = 0.1
dissolve_stagger = 0.1
dissolve_duration = 0.5
total_duration = 1.5
fps = 60

print("Letter positions and timing:")
print("-" * 60)

for i, char in enumerate(text):
    letter_start_time = stable_duration + i * dissolve_stagger
    letter_end_time = letter_start_time + dissolve_duration
    
    start_frame = int(letter_start_time / total_duration * fps * total_duration)
    end_frame = int(letter_end_time / total_duration * fps * total_duration)
    
    char_display = f"'{char}'" if char != ' ' else "'SPACE'"
    print(f"Index {i:2d}: {char_display:8s} | Start: {letter_start_time:.2f}s (frame {start_frame:3d}) | End: {letter_end_time:.2f}s (frame {end_frame:3d})")

print("\nSpace is at index 5")
print("Letters before space: H-E-L-L-O (indices 0-4)")
print("Letters after space: W-O-R-L-D (indices 6-10)")

# Calculate when the jump would occur
space_start = stable_duration + 5 * dissolve_stagger
w_start = stable_duration + 6 * dissolve_stagger
print(f"\nSpace starts dissolving at: {space_start:.2f}s")
print(f"W starts dissolving at: {w_start:.2f}s")
print(f"Gap between O and W dissolve: {(w_start - (stable_duration + 4 * dissolve_stagger)):.2f}s")
print(f"Normal gap between letters: {dissolve_stagger:.2f}s")