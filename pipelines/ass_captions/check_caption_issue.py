#!/usr/bin/env python3
"""Debug why captions aren't visible despite threshold fix"""

import cv2
import numpy as np
import json

# Load transcript to see what should be visible
with open('../../uploads/assets/videos/ai_math1/transcript_enriched.json', 'r') as f:
    data = json.load(f)

# Find phrases active at t=5s and t=10s
for target_time in [5.0, 10.0]:
    print(f"\n=== At t={target_time}s ===")
    active_phrases = []
    for phrase in data['phrases']:
        if phrase['start_time'] <= target_time <= phrase['end_time']:
            active_phrases.append(phrase)
            print(f"Should show: '{phrase['text'][:40]}...'")
            print(f"  Duration: {phrase['start_time']:.1f}-{phrase['end_time']:.1f}s")
            print(f"  Position: {phrase.get('position', 'unknown')}")
    
    if not active_phrases:
        print("No phrases active at this time")

# Check if issue is opacity/transparency
print("\n=== Checking pixel values ===")
fixed = cv2.imread('/tmp/fixed_5s.png')
orig = cv2.imread('/tmp/orig_5s.png')

# Look for ANY text-like patterns
diff = cv2.absdiff(fixed, orig)
gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

# Find areas with changes
changed_pixels = gray_diff > 5
num_changed = np.sum(changed_pixels)
print(f"Pixels with changes > 5: {num_changed} ({num_changed/gray_diff.size*100:.3f}%)")

# Check if changes form text-like patterns
if num_changed > 1000:
    # Find contours of changed areas
    contours, _ = cv2.findContours(changed_pixels.astype(np.uint8), 
                                   cv2.RETR_EXTERNAL, 
                                   cv2.CHAIN_APPROX_SIMPLE)
    print(f"Number of changed regions: {len(contours)}")
    
    # Check sizes of changed regions
    areas = [cv2.contourArea(c) for c in contours]
    if areas:
        print(f"Largest changed area: {max(areas)} pixels")
        print(f"Average changed area: {np.mean(areas):.1f} pixels")
