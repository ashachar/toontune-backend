#!/usr/bin/env python3
"""Create a simple test video with a person (solid shape) for testing occlusion."""

import cv2
import numpy as np

# Create a 2-second test video with a "person" (rectangle) moving
width, height = 1280, 720
fps = 30
duration = 2
frames = fps * duration

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('test_person.mp4', fourcc, fps, (width, height))

for i in range(frames):
    # Create gray background
    frame = np.full((height, width, 3), 200, dtype=np.uint8)
    
    # Draw moving "person" (blue rectangle)
    person_x = 400 + int(200 * np.sin(i * 0.1))  # Oscillating movement
    person_y = 200
    person_w = 200
    person_h = 400
    
    # Draw person
    cv2.rectangle(frame, (person_x, person_y), 
                  (person_x + person_w, person_y + person_h), 
                  (100, 50, 0), -1)  # Dark blue fill
    
    # Add some details to make it more person-like
    # Head
    head_center = (person_x + person_w // 2, person_y + 50)
    cv2.circle(frame, head_center, 40, (150, 100, 50), -1)
    
    out.write(frame)

out.release()
print("✅ Created test_person.mp4")