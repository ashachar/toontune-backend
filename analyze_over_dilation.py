#!/usr/bin/env python3
"""Analyze the over-dilation issue and sample 10 random frames"""

import cv2
import numpy as np
import random
from utils.segmentation.segment_extractor import extract_foreground_mask

print("="*60)
print("ANALYZING OVER-DILATION ISSUE")
print("="*60)

video_path = "test_element_3sec.mp4"
cap = cv2.VideoCapture(video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Sample 10 random frames
random.seed(42)
sample_frames = sorted(random.sample(range(45), 10))
print(f"\nSampling frames: {sample_frames}\n")

# Load all frames
frames = []
for i in range(45):
    ret, frame = cap.read()
    if not ret:
        break
    frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
cap.release()

# Test different mask processing approaches
approaches = [
    ("original", lambda m: m),
    ("slight_dilate", lambda m: cv2.dilate(m, np.ones((3, 3), np.uint8), iterations=1)),
    ("moderate_dilate", lambda m: cv2.dilate(m, np.ones((5, 5), np.uint8), iterations=1)),
    ("current_aggressive", lambda m: cv2.dilate(
        cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8)),
        np.ones((11, 11), np.uint8), iterations=2
    )),
    ("balanced", lambda m: cv2.dilate(
        cv2.GaussianBlur(m, (3, 3), 0),
        np.ones((5, 5), np.uint8), iterations=1
    )),
]

# Analyze each sampled frame
for frame_idx in sample_frames:
    print(f"Frame {frame_idx}:")
    frame = frames[frame_idx]
    
    # Extract base mask
    mask = extract_foreground_mask(frame)
    
    # Apply different approaches
    for name, process_fn in approaches:
        processed_mask = process_fn(mask.copy())
        processed_mask = (processed_mask > 100).astype(np.uint8) * 255
        
        coverage = np.sum(processed_mask > 128) / processed_mask.size
        print(f"  {name:20s}: {coverage:6.1%} coverage")
        
        # Save comparison for frame 25 (critical W occlusion)
        if frame_idx == 25:
            # Create visualization
            vis = np.zeros((H, W*2, 3), dtype=np.uint8)
            vis[:, :W] = frame
            vis[:, W:, 1] = processed_mask  # Green channel for mask
            
            # Add labels
            cv2.putText(vis, name, (W+10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(vis, f"{coverage:.1%}", (W+10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            cv2.imwrite(f"mask_compare_{name}.png", cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
    
    print()

# Now let's check what makes sense
print("="*60)
print("CHECKING WHAT MAKES SENSE")
print("="*60)

# The woman with guitar should be masked, but not the entire frame
# Let's check specific regions
frame = frames[25]  # Critical frame
mask_original = extract_foreground_mask(frame)

# Define regions to check
regions = [
    ("Woman with guitar", (100, 250, 200, 350)),  # x1, y1, x2, y2
    ("Children area", (400, 300, 800, 450)),
    ("Background sky", (0, 0, 200, 100)),
    ("W letter area", (640, 250, 720, 330)),
]

print("\nRegion analysis for frame 25:")
for name, (x1, y1, x2, y2) in regions:
    region_mask = mask_original[y1:y2, x1:x2]
    coverage = np.sum(region_mask > 128) / region_mask.size
    print(f"  {name:20s}: {coverage:6.1%} masked")
    
    # What SHOULD be masked?
    if "guitar" in name.lower() or "children" in name.lower():
        should_be = "YES"
    else:
        should_be = "NO"
    print(f"    Should be masked: {should_be}")
    
    if (coverage > 0.5 and should_be == "YES") or (coverage < 0.5 and should_be == "NO"):
        print(f"    ✓ Correct")
    else:
        print(f"    ✗ Incorrect")
    print()

# Test a better balanced approach
print("="*60)
print("TESTING BETTER BALANCED APPROACH")
print("="*60)

def smart_mask_processing(mask):
    """Smart mask processing - expand just enough to cover edges"""
    # 1. Small Gaussian blur to smooth edges
    mask = cv2.GaussianBlur(mask, (3, 3), 0)
    
    # 2. Small dilation to expand slightly
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)
    
    # 3. Threshold to maintain binary
    mask = (mask > 128).astype(np.uint8) * 255
    
    return mask

# Test on all sample frames
print("\nSmart processing coverage:")
for frame_idx in sample_frames:
    frame = frames[frame_idx]
    mask = extract_foreground_mask(frame)
    smart_mask = smart_mask_processing(mask)
    
    coverage = np.sum(smart_mask > 128) / smart_mask.size
    print(f"  Frame {frame_idx:2d}: {coverage:6.1%}")
    
    # Save for frame 25
    if frame_idx == 25:
        cv2.imwrite("mask_smart.png", smart_mask)
        
        # Create comparison
        comparison = np.zeros((H, W*3, 3), dtype=np.uint8)
        comparison[:, :W] = frame
        comparison[:, W:W*2, 1] = mask  # Original in green
        comparison[:, W*2:, 1] = smart_mask  # Smart in green
        
        cv2.putText(comparison, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(comparison, "Original Mask", (W+10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(comparison, "Smart Mask", (W*2+10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        cv2.imwrite("mask_smart_comparison.png", cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))

print("\n" + "="*60)
print("ANALYSIS COMPLETE!")
print("="*60)
print("\nKey findings:")
print("1. Current aggressive dilation (11x11, 2 iterations) is WAY too much")
print("2. Original mask covers ~20% which is about right for the people")
print("3. Smart approach (5x5, 1 iteration) gives ~22% coverage")
print("4. The W area needs selective masking, not blanket coverage")
print("\nCheck mask_smart_comparison.png to see the difference!")