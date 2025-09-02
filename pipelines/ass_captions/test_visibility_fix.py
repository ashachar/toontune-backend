#!/usr/bin/env python3
"""Quick test to verify visibility threshold fix"""

import cv2
import numpy as np
import sys

# Test parameters
green = np.array([154, 254, 119])  # Correct BGR green
visibility_threshold_old = 0.9
visibility_threshold_new = 0.4

# Load a sample mask frame
cap = cv2.VideoCapture('../../uploads/assets/videos/ai_math1/ai_math1_rvm_mask.mp4')
cap.set(cv2.CAP_PROP_POS_MSEC, 5000)  # 5 seconds in
ret, mask_frame = cap.read()
cap.release()

if not ret:
    print("Error: Could not read mask frame")
    sys.exit(1)

h, w = mask_frame.shape[:2]

# Extract person mask
diff = np.abs(mask_frame.astype(np.int16) - green.astype(np.int16))
is_green = np.all(diff <= 25, axis=2)
person_mask = (~is_green).astype(np.uint8)

# Simulate text placement
test_phrases = [
    ("invented a new calculus operator", 540),  # bottom
    ("AI created new math", 180),  # top
]

print("Visibility analysis with different thresholds:")
print("=" * 60)

for text, y_pos in test_phrases:
    # Calculate text bbox
    font_size = int(48 * 1.3)
    text_width = len(text) * int(font_size * 0.6)
    x = (w - text_width) // 2
    text_bbox = (x, y_pos - font_size, x + text_width, y_pos + int(font_size * 0.5))
    
    # Clip to frame bounds
    x1, y1, x2, y2 = text_bbox
    x1 = max(0, min(x1, w))
    x2 = max(0, min(x2, w))
    y1 = max(0, min(y1, h))
    y2 = max(0, min(y2, h))
    
    # Calculate visibility
    text_region = person_mask[y1:y2, x1:x2]
    total_pixels = text_region.size
    if total_pixels > 0:
        person_pixels = np.sum(text_region == 1)
        visibility = 1.0 - (person_pixels / total_pixels)
    else:
        visibility = 1.0
    
    print(f"\nText: '{text[:30]}...'")
    print(f"  Position: Y={y_pos} ({'bottom' if y_pos > 400 else 'top'})")
    print(f"  Visibility: {visibility*100:.1f}%")
    print(f"  Old threshold (90%): {'FRONT' if visibility >= visibility_threshold_old else 'BEHIND (hidden)'}")
    print(f"  New threshold (40%): {'FRONT' if visibility >= visibility_threshold_new else 'BEHIND (hidden)'}")

print("\n" + "=" * 60)
print("Summary:")
print("  With old 90% threshold: Most text goes BEHIND (hidden)")
print("  With new 40% threshold: Text with >40% visibility stays FRONT (visible)")
