"""
Check the original video brightness to see if jumps come from the source
"""

import cv2
import numpy as np

def check_original_brightness():
    """Check brightness of original video frames"""
    
    print("Checking original video brightness (no animation)")
    print("=" * 60)
    
    # Load original video
    cap = cv2.VideoCapture("uploads/assets/videos/ai_math1.mp4")
    
    if not cap.isOpened():
        print("Error: Could not open video")
        return
    
    # Region of interest (where text would appear)
    roi_x1, roi_y1 = 400, 320
    roi_x2, roi_y2 = 880, 400
    
    print("Frame | Mean Brightness | Notes")
    print("-" * 40)
    
    # Check frames 60-75
    for frame_num in range(60, 76):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Extract ROI
        roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray_roi)
        
        notes = ""
        if frame_num == 65:
            notes = "← Before jump"
        elif frame_num == 66:
            notes = "← Big jump up!"
        elif frame_num == 69:
            notes = "← Big drop!"
        
        print(f"{frame_num:3d} | {mean_brightness:6.1f} | {notes}")
    
    cap.release()
    
    print("\n" + "=" * 60)
    print("CONCLUSION:")
    print("If the original video also has brightness jumps,")
    print("then the 'flashing' is from the source video, not the animation!")

if __name__ == "__main__":
    check_original_brightness()