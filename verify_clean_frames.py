#!/usr/bin/env python3
"""Verify if frames after 216 are truly clean."""

import cv2
import numpy as np

cap = cv2.VideoCapture("hello_world_fixed.mp4")

# Get the very first frame (before text appears)
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
ret, first_frame = cap.read()

# Get frame 220 (should be clean if fix worked)
cap.set(cv2.CAP_PROP_POS_FRAMES, 220)
ret, frame_220 = cap.read()

# Get frame 250 
cap.set(cv2.CAP_PROP_POS_FRAMES, 250)
ret, frame_250 = cap.read()

# Get frame 300 (very late, should definitely be clean)
cap.set(cv2.CAP_PROP_POS_FRAMES, 300)
ret, frame_300 = cap.read()

cap.release()

print("Comparing frames to detect text presence...")
print()

# Compare frame 220 to first frame
diff_220 = cv2.absdiff(frame_220, first_frame)
max_diff_220 = np.max(diff_220)
mean_diff_220 = np.mean(diff_220)

print(f"Frame 220 vs Frame 0:")
print(f"  Max difference: {max_diff_220}")
print(f"  Mean difference: {mean_diff_220:.2f}")

# Compare frame 250 to first frame
diff_250 = cv2.absdiff(frame_250, first_frame)
max_diff_250 = np.max(diff_250)
mean_diff_250 = np.mean(diff_250)

print(f"Frame 250 vs Frame 0:")
print(f"  Max difference: {max_diff_250}")
print(f"  Mean difference: {mean_diff_250:.2f}")

# Compare frame 300 to first frame
diff_300 = cv2.absdiff(frame_300, first_frame)
max_diff_300 = np.max(diff_300)
mean_diff_300 = np.mean(diff_300)

print(f"Frame 300 vs Frame 0:")
print(f"  Max difference: {max_diff_300}")
print(f"  Mean difference: {mean_diff_300:.2f}")

print()

# Check if frames 220 and 300 are identical (they should be if both are clean)
diff_220_300 = cv2.absdiff(frame_220, frame_300)
if np.max(diff_220_300) < 5:
    print("✓ Frames 220 and 300 are identical - fix is working!")
else:
    print("⚠️  Frames 220 and 300 differ - text might still be fading")
    print(f"   Max difference between 220 and 300: {np.max(diff_220_300)}")

# Save comparison images
comparison = np.hstack([first_frame, frame_220, frame_250, frame_300])
cv2.putText(comparison, "Frame 0", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
cv2.putText(comparison, "Frame 220", (50 + 1280, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
cv2.putText(comparison, "Frame 250", (50 + 2560, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
cv2.putText(comparison, "Frame 300", (50 + 3840, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

# Scale down for visibility
h, w = comparison.shape[:2]
comparison_small = cv2.resize(comparison, (w//4, h//4))
cv2.imwrite("frame_comparison.png", comparison_small)
print("\nSaved frame_comparison.png for visual inspection")