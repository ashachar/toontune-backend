#!/usr/bin/env python3
"""
Verify that the dissolve effect is working properly by extracting key frames.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

def extract_frames(video_path, frame_indices):
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
    # Key frame indices during dissolve phase
    # Frame 90: Start of dissolve
    # Frame 95: Early dissolve
    # Frame 100: Mid dissolve
    # Frame 110: Late dissolve
    # Frame 120: Near end of dissolve
    # Frame 140: End of dissolve
    key_indices = [90, 95, 100, 110, 120, 140]
    
    # Extract frames from refactored version
    frames = extract_frames("start_animation_refactored.mp4", key_indices)
    
    # Create comparison plot
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    titles = [
        "Frame 90: Dissolve Start",
        "Frame 95: Early Dissolve",
        "Frame 100: Mid Dissolve",
        "Frame 110: Late Dissolve", 
        "Frame 120: Near End",
        "Frame 140: Dissolve Complete"
    ]
    
    for i, (frame, title) in enumerate(zip(frames, titles)):
        axes[i].imshow(frame)
        axes[i].set_title(title)
        axes[i].axis('off')
    
    plt.suptitle("Dissolve Effect Verification - Fixed Animation", fontsize=14)
    plt.tight_layout()
    plt.savefig("dissolve_effect_verification.png", dpi=150, bbox_inches='tight')
    print("Saved dissolve effect verification to dissolve_effect_verification.png")
    
    # Analyze dissolve progression
    print("\nDissolve Effect Analysis:")
    print("-" * 40)
    
    if len(frames) >= 2:
        # Compare brightness in text region to verify gradual fade
        h, w = frames[0].shape[:2]
        text_region_y1 = int(h * 0.35)
        text_region_y2 = int(h * 0.55)
        text_region_x1 = int(w * 0.35)
        text_region_x2 = int(w * 0.65)
        
        brightnesses = []
        for i, frame in enumerate(frames):
            region = frame[text_region_y1:text_region_y2, text_region_x1:text_region_x2]
            avg_brightness = np.mean(region)
            brightnesses.append(avg_brightness)
            print(f"Frame {key_indices[i]}: Avg brightness = {avg_brightness:.1f}")
        
        # Check if brightness decreases gradually (indicating fade)
        is_gradual = all(brightnesses[i] >= brightnesses[i+1] - 5 for i in range(len(brightnesses)-1))
        
        print(f"\n{'✓' if is_gradual else '✗'} Dissolve is {'gradual' if is_gradual else 'not gradual'}")
        
        # Check for smooth transition (no jumps)
        max_jump = max(abs(brightnesses[i] - brightnesses[i+1]) for i in range(len(brightnesses)-1))
        print(f"Maximum brightness jump: {max_jump:.1f}")
        print(f"{'✓ Smooth' if max_jump < 20 else '✗ Abrupt'} transition detected")

if __name__ == "__main__":
    main()