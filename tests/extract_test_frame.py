#!/usr/bin/env python3
"""Extract a single frame for testing."""

import cv2
from pathlib import Path

video_path = "uploads/assets/videos/do_re_mi/scenes/original/scene_001.mp4"
timestamp = 24.0

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_number = int(timestamp * fps)
cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
ret, frame = cap.read()

if ret:
    output_path = Path("tests/test_frame_t24.jpg")
    cv2.imwrite(str(output_path), frame)
    print(f"Frame saved to {output_path}")
    print(f"Frame shape: {frame.shape}")
    
    # Also save a smaller version for quick viewing
    small = cv2.resize(frame, (584, 263))
    cv2.imwrite("tests/test_frame_t24_small.jpg", small)
    print("Small version saved to tests/test_frame_t24_small.jpg")
else:
    print("Failed to extract frame")

cap.release()