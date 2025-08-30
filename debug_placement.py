#!/usr/bin/env python3
"""Debug why face-aware positioning isn't working"""

import cv2
import numpy as np
from utils.text_placement.stripe_layout_manager import StripeLayoutManager

# Load sample frame
frame = cv2.imread('./outputs/frame_check.png')

# Create layout manager
layout_manager = StripeLayoutManager()

# Create sample phrase
phrases = [{
    'phrase': 'Yes,',
    'importance': 0.5,
    'layout_priority': 1
}]

# Create sample masks (simplified)
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 50, 150)
foreground_masks = [edges]

# Test layout with sample frames
sample_frames = [frame]  # Pass the actual frame for face detection

# Call layout_scene_phrases
placements = layout_manager.layout_scene_phrases(phrases, foreground_masks, sample_frames)

print("Placement results:")
for p in placements:
    print(f"  Phrase: '{p.phrase}'")
    print(f"  Position: {p.position}")
    print(f"  Is behind: {p.is_behind}")
    print(f"  Visibility: {p.visibility_score:.2f}")