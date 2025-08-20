#!/usr/bin/env python3
"""
Debug what WordDissolve is doing at frame 0.
"""

import os
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.animations.word_dissolve import WordDissolve


def debug_word_dissolve():
    """Debug WordDissolve at frame 0."""
    
    width = 1168
    height = 526
    center_position = (width // 2, int(height * 0.45))
    font_size = 147
    
    # Create dummy image
    dummy_img = Image.new('RGB', (100, 100), (0, 0, 0))
    dummy_img.save('/tmp/dummy.png')
    
    # Create WordDissolve with no randomization
    word_dissolver = WordDissolve(
        element_path="/tmp/dummy.png",
        background_path="/tmp/dummy.png",
        position=center_position,
        word="START",
        font_size=font_size,
        text_color=(255, 220, 0),
        dissolve_duration=0.67,
        dissolve_stagger=0.33,
        float_distance=30,
        randomize_order=False,  # Important: no randomization
        maintain_kerning=True,
        center_position=center_position,
        fps=30
    )
    
    print("WordDissolve configuration:")
    print(f"  Word: '{word_dissolver.word}'")
    print(f"  Font size: {word_dissolver.font_size}")
    print(f"  Center position: {word_dissolver.center_position}")
    print(f"  Dissolve frames: {word_dissolver.dissolve_frames}")
    print(f"  Stagger frames: {word_dissolver.stagger_frames}")
    print()
    
    print("Letter dissolvers:")
    for i, dissolver_info in enumerate(word_dissolver.letter_dissolvers):
        print(f"  {i}: Letter '{dissolver_info['letter']}' - "
              f"index={dissolver_info['index']}, "
              f"start_frame={dissolver_info['start_frame']}, "
              f"dissolve_order={dissolver_info['dissolve_order']}")
    print()
    
    # Simulate what happens at frame 0
    frame_idx = 0
    non_dissolving_text = ""
    dissolving_indices = []
    
    for i, letter in enumerate(word_dissolver.word):
        dissolver_info = next(d for d in word_dissolver.letter_dissolvers if d['index'] == i)
        start_frame = dissolver_info['start_frame']
        
        print(f"Letter {i} ('{letter}'): start_frame={start_frame}")
        
        if frame_idx < start_frame:
            print(f"  -> Not dissolving yet, adding to non_dissolving_text")
            non_dissolving_text += letter
        elif frame_idx < start_frame + word_dissolver.dissolve_frames:
            print(f"  -> Currently dissolving")
            dissolving_indices.append(i)
        else:
            print(f"  -> Fully dissolved")
    
    print()
    print(f"At frame {frame_idx}:")
    print(f"  Non-dissolving text: '{non_dissolving_text}'")
    print(f"  Dissolving indices: {dissolving_indices}")
    
    # Check if first letter starts at frame 0
    first_letter_start = word_dissolver.letter_dissolvers[0]['start_frame']
    print(f"\nFirst letter starts dissolving at frame: {first_letter_start}")
    
    if first_letter_start == 0:
        print("⚠️ PROBLEM: First letter starts dissolving immediately at frame 0!")
        print("This means at frame 0, some letters might already be dissolving.")


if __name__ == "__main__":
    debug_word_dissolve()