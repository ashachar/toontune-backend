#!/usr/bin/env python3
"""Detect black margins in Runway video"""

import cv2
import numpy as np

# Load frame from Runway video
video_path = "uploads/assets/runway_experiment/runway_act_two_output.mp4"
cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
cap.release()

if ret:
    h, w = frame.shape[:2]
    print(f"Frame size: {w}x{h}")
    
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Find left margin (scan from left until non-black)
    left_margin = 0
    for x in range(w):
        if np.any(gray[:, x] > 20):  # Threshold for "not black"
            left_margin = x
            break
    
    # Find right margin (scan from right until non-black)
    right_margin = w
    for x in range(w-1, -1, -1):
        if np.any(gray[:, x] > 20):
            right_margin = x + 1
            break
    
    # Find top margin
    top_margin = 0
    for y in range(h):
        if np.any(gray[y, :] > 20):
            top_margin = y
            break
    
    # Find bottom margin
    bottom_margin = h
    for y in range(h-1, -1, -1):
        if np.any(gray[y, :] > 20):
            bottom_margin = y + 1
            break
    
    print(f"Black margins detected:")
    print(f"  Left: {left_margin} pixels")
    print(f"  Right: {w - right_margin} pixels")
    print(f"  Top: {top_margin} pixels")
    print(f"  Bottom: {h - bottom_margin} pixels")
    
    content_width = right_margin - left_margin
    content_height = bottom_margin - top_margin
    print(f"\nActual content area: {content_width}x{content_height}")
    print(f"Content starts at: ({left_margin}, {top_margin})")
    
    # Compare with original
    print(f"\nOriginal video: 1280x720")
    print(f"Runway with margins: {w}x{h}")
    print(f"Runway content only: {content_width}x{content_height}")
    
    # Calculate what needs to be done
    scale_factor = max(1280 / content_width, 720 / content_height)
    print(f"\nTo cover 1280x720, content needs to be scaled by: {scale_factor:.3f}x")
    scaled_w = int(content_width * scale_factor)
    scaled_h = int(content_height * scale_factor)
    print(f"Scaled content size: {scaled_w}x{scaled_h}")
    print(f"Then crop to: 1280x720")