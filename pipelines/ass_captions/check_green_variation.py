#!/usr/bin/env python3
import cv2
import numpy as np

# Load mask video
cap_mask = cv2.VideoCapture("../../uploads/assets/videos/ai_math1/ai_math1_rvm_mask.mp4")

green_target = np.array([154, 254, 119], dtype=np.uint8)
all_greens = []

print("Checking green color variation across frames...")

# Check first 30 frames
for frame_idx in range(30):
    ret, mask_frame = cap_mask.read()
    if not ret:
        break
    
    # Get all unique colors in the frame
    unique_colors = np.unique(mask_frame.reshape(-1, 3), axis=0)
    
    # Find colors close to green
    for color in unique_colors:
        diff = np.abs(color.astype(int) - green_target.astype(int))
        if np.max(diff) <= 10:  # Within 10 of each channel
            all_greens.append(color)

cap_mask.release()

# Analyze the green variations
all_greens = np.array(all_greens)
unique_greens = np.unique(all_greens, axis=0)

print(f"\nFound {len(unique_greens)} unique green-ish colors across frames:")
for color in unique_greens[:20]:  # Show first 20
    b, g, r = color
    diff = np.abs(color.astype(int) - green_target.astype(int))
    max_diff = np.max(diff)
    print(f"  BGR [{b:3}, {g:3}, {r:3}] - max diff: {max_diff}")

if len(unique_greens) > 20:
    print(f"  ... and {len(unique_greens)-20} more")

# Check specific frame at 2.5 seconds where text appears
cap_mask = cv2.VideoCapture("../../uploads/assets/videos/ai_math1/ai_math1_rvm_mask.mp4") 
cap_mask.set(cv2.CAP_PROP_POS_MSEC, 2500)
ret, mask_frame = cap_mask.read()

if ret:
    print("\n" + "="*60)
    print("Frame at 2.5s (where text appears):")
    
    # Check if exact green exists
    exact_match = np.all(mask_frame == green_target, axis=2)
    print(f"Pixels with EXACT green [154, 254, 119]: {np.sum(exact_match)}")
    
    # Check with tolerance
    for tolerance in [1, 2, 3, 5, 10]:
        # Check if within tolerance of green
        diff = np.abs(mask_frame.astype(int) - green_target.astype(int))
        within_tolerance = np.all(diff <= tolerance, axis=2)
        count = np.sum(within_tolerance)
        percent = 100 * count / within_tolerance.size
        print(f"Pixels within ±{tolerance:2} of green: {count:7} ({percent:5.1f}%)")

cap_mask.release()

print("\n🔴 CONCLUSION: Green color varies due to video compression!")
print("Need to use tolerance-based matching, not exact color match.")