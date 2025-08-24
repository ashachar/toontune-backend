#!/usr/bin/env python3
"""Verify the timing after excluding spaces from dissolve order."""

text = "HELLO WORLD"
stable_duration = 0.1
dissolve_stagger = 0.1
dissolve_duration = 0.5
total_duration = 1.5
fps = 60

# New dissolve order excludes space at index 5
dissolve_order = [0, 1, 2, 3, 4, 6, 7, 8, 9, 10]

print("Letter positions and timing (AFTER FIX - spaces excluded):")
print("-" * 60)

for order_idx, letter_idx in enumerate(dissolve_order):
    char = text[letter_idx]
    letter_start_time = stable_duration + order_idx * dissolve_stagger
    letter_end_time = letter_start_time + dissolve_duration
    
    start_frame = int(letter_start_time / total_duration * fps * total_duration)
    end_frame = int(letter_end_time / total_duration * fps * total_duration)
    
    char_display = f"'{char}'"
    print(f"Order {order_idx:2d}, Index {letter_idx:2d}: {char_display:8s} | Start: {letter_start_time:.2f}s (frame {start_frame:3d}) | End: {letter_end_time:.2f}s (frame {end_frame:3d})")

print("\nKey observations:")
print("- Space (index 5) is NOT in the dissolve order")
print("- 'O' of HELLO (index 4) dissolves at position 4")
print("- 'W' of WORLD (index 6) dissolves at position 5")
print("- Gap between 'O' and 'W' is now 0.10s (normal stagger)")
print("- No more double-gap jump!")