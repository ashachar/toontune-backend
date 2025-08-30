#!/usr/bin/env python3
"""Test face detection on the actual frame"""

import cv2
import numpy as np
from utils.text_placement.face_aware_placement import FaceDetector

# Load the frame
frame = cv2.imread('./outputs/frame_check.png')

# Create detector and detect faces
detector = FaceDetector()
faces = detector.detect_faces(frame)

print(f"Detected {len(faces)} faces")
for i, face in enumerate(faces):
    print(f"Face {i}: x={face.x}, y={face.y}, width={face.width}, height={face.height}")
    print(f"  Left edge: {face.left}, Right edge: {face.right}")
    
    # Draw rectangle on image for visualization
    cv2.rectangle(frame, (face.left, face.top), (face.right, face.bottom), (0, 255, 0), 2)

# Test text positioning
text_width = 100  # Approximate width of "Yes,"
safe_positions = detector.get_safe_x_positions(faces, frame.shape[1], text_width)
print(f"\nSafe positions for text (width={text_width}):")
for pos, score in safe_positions[:3]:
    print(f"  x={pos}, score={score:.2f}")

# Save visualization
cv2.imwrite('./outputs/face_detection_test.png', frame)
print("\nVisualization saved to ./outputs/face_detection_test.png")