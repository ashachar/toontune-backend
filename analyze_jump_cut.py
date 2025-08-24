#!/usr/bin/env python3
"""Analyze debug frames to detect jump-cuts between letters."""

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load debug frames
frame_numbers = [10, 15, 20, 25, 30, 35]
frames = []

for num in frame_numbers:
    path = f"debug_dissolve_frame_{num:03d}.png"
    frame = cv2.imread(path)
    if frame is not None:
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        print(f"Loaded frame {num}")
    else:
        print(f"Could not load frame {num}")

if len(frames) >= 2:
    # Create a comparison image showing all frames
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    
    for i, (frame, num) in enumerate(zip(frames, frame_numbers)):
        axes[i].imshow(frame)
        axes[i].set_title(f"Frame {num} - Letters {'H-E-L-L-O'[:min(num//6, 10)]}")
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('jump_cut_analysis.png', dpi=150, bbox_inches='tight')
    print("Saved analysis to jump_cut_analysis.png")
    
    # Also create a focused view on the text area
    fig2, axes2 = plt.subplots(1, len(frames), figsize=(20, 4))
    
    for i, (frame, num) in enumerate(zip(frames, frame_numbers)):
        # Crop to text area (middle of frame)
        h, w = frame.shape[:2]
        text_area = frame[h//3:2*h//3, w//4:3*w//4]
        axes2[i].imshow(text_area)
        axes2[i].set_title(f"Frame {num}")
        axes2[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('jump_cut_text_focus.png', dpi=150, bbox_inches='tight')
    print("Saved text focus to jump_cut_text_focus.png")