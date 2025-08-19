#!/usr/bin/env python3
"""
View the CoTracker3 results
"""

import cv2
import numpy as np
import os

def view_tracking_result():
    """Display the tracking results side by side"""
    
    # Paths
    original = "tests/tracking_test.mov"
    tracked = "tests/tracking_test_tracked.mp4"
    
    if not os.path.exists(tracked):
        print("❌ Tracked video not found. Run test_cotracker_pipeline.py first!")
        return
    
    print("🎬 Displaying tracking results...")
    print("  Press 'q' to quit")
    print("  Press 'space' to pause/resume")
    print("  Press 'r' to restart")
    
    # Open both videos
    cap_orig = cv2.VideoCapture(original)
    cap_tracked = cv2.VideoCapture(tracked)
    
    fps = cap_orig.get(cv2.CAP_PROP_FPS)
    delay = int(1000 / fps)
    
    paused = False
    
    while True:
        if not paused:
            ret1, frame1 = cap_orig.read()
            ret2, frame2 = cap_tracked.read()
            
            if not ret1 or not ret2:
                # Restart videos
                cap_orig.set(cv2.CAP_PROP_POS_FRAMES, 0)
                cap_tracked.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
        
        # Create side-by-side view
        h1, w1 = frame1.shape[:2]
        h2, w2 = frame2.shape[:2]
        
        # Resize if needed to match heights
        if h1 != h2:
            scale = h1 / h2
            frame2 = cv2.resize(frame2, (int(w2 * scale), h1))
        
        # Concatenate horizontally
        combined = np.hstack([frame1, frame2])
        
        # Add labels
        cv2.putText(combined, "Original", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(combined, "Tracked", (w1 + 10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Display
        cv2.imshow("CoTracker3 Results - Press 'q' to quit", combined)
        
        # Handle keys
        key = cv2.waitKey(delay if not paused else 0) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord(' '):
            paused = not paused
        elif key == ord('r'):
            cap_orig.set(cv2.CAP_PROP_POS_FRAMES, 0)
            cap_tracked.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    cap_orig.release()
    cap_tracked.release()
    cv2.destroyAllWindows()
    
    print("✅ Done!")


if __name__ == "__main__":
    view_tracking_result()