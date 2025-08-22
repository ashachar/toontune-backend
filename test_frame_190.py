
import os
os.environ['FRAME_DISSOLVE_DEBUG'] = '1'

import cv2
import numpy as np
from rembg import remove, new_session
from utils.animations.text_behind_segment import TextBehindSegment
from utils.animations.word_dissolve import WordDissolve


import numpy as np

# Monkey patch to add debug
original_render = WordDissolve.render_word_frame

def debug_render(self, frame, frame_idx, mask=None):
    # Call original
    result = original_render(self, frame, frame_idx, mask)
    
    # Add debug at key frame
    if frame_idx == 190:
        print(f"[DEBUG_190] At frame 190, checking render logic")
        # This will be printed if we reach this frame
    
    return result

WordDissolve.render_word_frame = debug_render


# Run just the critical part
cap = cv2.VideoCapture("test_element_3sec.mp4")
fps = 60
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Quick test - just check frame 190 of WordDissolve
dissolve = WordDissolve(
    element_path="test_element_3sec.mp4",
    background_path="test_element_3sec.mp4",
    position=(width // 2, height // 2),
    word="HELLO WORLD",
    font_size=130,
    text_color=(255, 220, 0),
    stable_duration=0.1,
    dissolve_duration=1.0,
    dissolve_stagger=0.25,
    fps=fps,
    debug=True
)

# Get a frame
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
ret, frame = cap.read()
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# Test frame 190
result = dissolve.render_word_frame(frame_rgb, 190, mask=None)

# Check if result has yellow pixels
yellow_mask = (result[:,:,0] > 180) & (result[:,:,1] > 180) & (result[:,:,2] < 100)
yellow_pixels = np.sum(yellow_mask)
print(f"[DEBUG_190] Yellow pixels in result: {yellow_pixels}")

cap.release()
