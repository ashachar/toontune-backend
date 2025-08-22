#!/usr/bin/env python3
"""Test the actual dissolve process to understand where artifacts come from."""

import os
os.environ['FRAME_DISSOLVE_DEBUG'] = '1'

import cv2
import numpy as np
from utils.animations.word_dissolve import WordDissolve

# Create test video with just the W dissolving
def create_test_video():
    # Parameters
    width, height = 1280, 492
    fps = 60
    duration = 5  # seconds
    total_frames = fps * duration
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('w_dissolve_test.mp4', fourcc, fps, (width, height))
    
    # Create background (green field)
    background = np.ones((height, width, 3), dtype=np.uint8) * [50, 80, 50]
    
    # Create empty occlusion mask (no foreground objects)
    occlusion_mask = np.zeros((height, width), dtype=np.uint8)
    
    # Initialize dissolve animation for just "W"
    dissolve = WordDissolve(
        word="W",
        font_path="/System/Library/Fonts/Helvetica.ttc",
        font_size=200,
        position=(640, 250),  # Center of frame
        color=(255, 255, 0),  # Yellow
        stable_frames=30,      # 0.5 seconds stable
        dissolve_duration=120, # 2 seconds to dissolve
        stagger_frames=0,      # No stagger for single letter
        max_scale=1.5,
        debug=True
    )
    
    # Prepare the animation
    dissolve.prepare(width, height, None, occlusion_mask)
    
    print(f"Creating test video with W dissolving...")
    print(f"  Stable frames: 0-30")
    print(f"  Dissolve frames: 30-150")
    print(f"  Empty frames: 150-{total_frames}")
    
    # Generate frames
    for frame_idx in range(total_frames):
        # Start with background
        frame = background.copy()
        
        # Apply dissolve effect
        result = dissolve.apply(frame_idx, frame, occlusion_mask)
        
        # Write frame
        out.write(result)
        
        # Log key frames
        if frame_idx in [0, 30, 60, 90, 120, 150]:
            cv2.imwrite(f"w_test_frame_{frame_idx:03d}.png", result)
            print(f"  Frame {frame_idx}: Saved")
    
    out.release()
    print(f"Created w_dissolve_test.mp4")
    
    return 'w_dissolve_test.mp4'

# Create the test video
video_path = create_test_video()

# Extract frames around the dissolve to check for artifacts
print("\nExtracting frames to check for artifacts...")
cap = cv2.VideoCapture(video_path)

# Focus on the dissolve period
for frame_idx in [30, 45, 60, 75, 90, 105, 120, 135, 150]:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    if ret:
        # Check for non-background pixels in areas around W
        # W should be roughly centered at (640, 250)
        # Check region around it for artifacts
        roi_x1, roi_x2 = 540, 740
        roi_y1, roi_y2 = 150, 350
        
        roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]
        
        # Background is [50, 80, 50] (BGR)
        bg_color = np.array([50, 80, 50])
        
        # Find pixels that differ from background
        diff = np.abs(roi.astype(np.float32) - bg_color.astype(np.float32))
        diff_max = np.max(diff, axis=2)
        
        # Pixels that differ by more than 5 from background
        non_bg_mask = diff_max > 5
        non_bg_count = np.sum(non_bg_mask)
        
        # Check for gray pixels (not yellow, not background)
        # Yellow is high B+G, low R in BGR
        yellow_mask = (roi[:,:,1] > 200) & (roi[:,:,2] > 200) & (roi[:,:,0] < 100)
        
        # Gray would be roughly equal RGB values, different from background
        gray_mask = non_bg_mask & ~yellow_mask
        gray_count = np.sum(gray_mask)
        
        print(f"Frame {frame_idx}: {non_bg_count} non-bg pixels, "
              f"{gray_count} potential gray artifacts")
        
        if gray_count > 10:
            # Save visualization
            highlight = roi.copy()
            highlight[gray_mask] = [0, 0, 255]  # Red for gray artifacts
            cv2.imwrite(f"w_test_artifacts_{frame_idx:03d}.png", highlight)

cap.release()

print("\nTest complete. Check w_dissolve_test.mp4 and artifact images.")