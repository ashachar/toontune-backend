#!/usr/bin/env python3
"""
Test rendering on a completely blank background to ensure no double letters.
"""

import cv2
import numpy as np
from utils.animations.text_3d_motion import Text3DMotion

# Create motion animation
motion = Text3DMotion(
    text="AI DEMO",
    font_size=160,
    start_scale=1.5,  # Not too large
    resolution=(1280, 720),
    debug=False
)

# Create a BLANK frame (gray background for visibility)
blank_frame = np.ones((720, 1280, 3), dtype=np.uint8) * 128  # Gray

# Render frame 5 (middle of motion)
result = motion.generate_frame(5, blank_frame)

# Save it
cv2.imwrite("test_blank_frame.png", cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
print("Saved test_blank_frame.png")
print("This should show ONLY ONE set of letters, not doubled")