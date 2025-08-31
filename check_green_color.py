#!/usr/bin/env python3
"""Check the exact green screen color in Runway video"""

import cv2
import numpy as np

# Load first frame of Runway video
video_path = "uploads/assets/runway_experiment/runway_act_two_output.mp4"
cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
cap.release()

if ret:
    # Get a sample of green pixels from corners and edges
    h, w = frame.shape[:2]
    
    # Sample green pixels from multiple locations
    green_samples = []
    
    # Top corners
    green_samples.append(frame[10, 10])
    green_samples.append(frame[10, w-10])
    
    # Bottom corners  
    green_samples.append(frame[h-10, 10])
    green_samples.append(frame[h-10, w-10])
    
    # Middle edges
    green_samples.append(frame[h//2, 10])
    green_samples.append(frame[h//2, w-10])
    green_samples.append(frame[10, w//2])
    green_samples.append(frame[h-10, w//2])
    
    # Calculate average green color
    green_samples = np.array(green_samples)
    avg_green = np.mean(green_samples, axis=0).astype(int)
    
    print(f"Green screen color samples (BGR format):")
    for i, sample in enumerate(green_samples):
        print(f"  Sample {i+1}: B={sample[0]}, G={sample[1]}, R={sample[2]}")
    
    print(f"\nAverage green screen color (BGR): [{avg_green[0]}, {avg_green[1]}, {avg_green[2]}]")
    print(f"Average green screen color (RGB): [{avg_green[2]}, {avg_green[1]}, {avg_green[0]}]")
    
    # Also check in HSV for better chroma keying
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    green_hsv_samples = []
    for pos in [(10,10), (10,w-10), (h-10,10), (h-10,w-10)]:
        green_hsv_samples.append(hsv[pos[0], pos[1]])
    
    green_hsv_samples = np.array(green_hsv_samples)
    avg_hsv = np.mean(green_hsv_samples, axis=0).astype(int)
    print(f"\nAverage green in HSV: H={avg_hsv[0]}, S={avg_hsv[1]}, V={avg_hsv[2]}")
    
    # Save a sample for visual verification
    cv2.imwrite("uploads/assets/runway_experiment/runway_first_frame.png", frame)
    print("\nSaved first frame to runway_first_frame.png for verification")