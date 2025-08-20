#!/usr/bin/env python3
"""
Verify that the position fix is working by checking key frames.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

def extract_key_frames(video_path, frame_indices):
    """Extract specific frames from video."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
    
    cap.release()
    return frames

def main():
    # Key frame indices to check
    # Frame 89: Last frame of stable phase (TextBehindSegment)
    # Frame 90: First frame of dissolve phase (WordDissolve)
    # Frame 100: During dissolve
    key_indices = [89, 90, 100]
    
    # Extract frames from refactored version
    refactored_frames = extract_key_frames(
        "start_animation_refactored.mp4",
        key_indices
    )
    
    # Create comparison plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    titles = [
        "Frame 89: End of Stable Phase",
        "Frame 90: Start of Dissolve Phase",
        "Frame 100: During Dissolve"
    ]
    
    for i, (frame, title) in enumerate(zip(refactored_frames, titles)):
        axes[i].imshow(frame)
        axes[i].set_title(title)
        axes[i].axis('off')
    
    plt.suptitle("Position Continuity Check - Refactored Animation", fontsize=14)
    plt.tight_layout()
    plt.savefig("position_continuity_check.png", dpi=150, bbox_inches='tight')
    print("Saved position continuity check to position_continuity_check.png")
    
    # Check pixel differences between transition frames
    if len(refactored_frames) >= 2:
        frame_89 = refactored_frames[0]
        frame_90 = refactored_frames[1]
        
        # Calculate difference in text region (approximate)
        h, w = frame_89.shape[:2]
        text_region_y1 = int(h * 0.35)
        text_region_y2 = int(h * 0.55)
        
        region_89 = frame_89[text_region_y1:text_region_y2, :]
        region_90 = frame_90[text_region_y1:text_region_y2, :]
        
        diff = np.abs(region_89.astype(float) - region_90.astype(float))
        mean_diff = np.mean(diff)
        
        print(f"\nText region analysis:")
        print(f"  Mean pixel difference between frames 89-90: {mean_diff:.2f}")
        print(f"  {'✓ Smooth transition' if mean_diff < 10 else '✗ Jump detected'}")

if __name__ == "__main__":
    main()