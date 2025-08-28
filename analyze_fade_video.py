"""
Analyze the fade debug video for flashing by checking brightness changes
"""

import cv2
import numpy as np

def analyze_video_for_flashing(video_path):
    """Check for brightness fluctuations that indicate flashing"""
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error opening {video_path}")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Analyzing {video_path}")
    print(f"FPS: {fps}, Total frames: {total_frames}")
    print("=" * 60)
    
    # Store brightness values for text region
    brightness_history = []
    
    # Region of interest (where text appears)
    roi_x1, roi_y1 = 400, 320
    roi_x2, roi_y2 = 880, 400
    
    frame_num = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Extract ROI
        roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]
        
        # Convert to grayscale and calculate mean brightness
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray_roi)
        brightness_history.append(mean_brightness)
        
        # Check for sudden changes every 5 frames
        if frame_num > 0 and frame_num % 5 == 0:
            # Look at last 5 frames
            recent = brightness_history[-5:]
            if len(recent) >= 2:
                # Calculate frame-to-frame differences
                diffs = [abs(recent[i+1] - recent[i]) for i in range(len(recent)-1)]
                max_diff = max(diffs)
                
                if max_diff > 10:  # Threshold for significant change
                    print(f"Frame {frame_num}: Large brightness change detected: {max_diff:.1f}")
                    print(f"  Recent brightness values: {[f'{b:.1f}' for b in recent]}")
        
        frame_num += 1
    
    cap.release()
    
    # Analyze overall pattern
    print("\n" + "=" * 60)
    print("BRIGHTNESS ANALYSIS SUMMARY")
    print("=" * 60)
    
    # Check for oscillations
    if len(brightness_history) > 10:
        # Look for periodic changes
        oscillations = 0
        direction_changes = 0
        prev_direction = None
        
        for i in range(1, len(brightness_history)):
            diff = brightness_history[i] - brightness_history[i-1]
            
            if abs(diff) > 5:  # Significant change
                oscillations += 1
                
            # Track direction changes
            if diff > 0:
                direction = "up"
            elif diff < 0:
                direction = "down"
            else:
                direction = "flat"
            
            if prev_direction and direction != "flat" and direction != prev_direction:
                direction_changes += 1
            
            if direction != "flat":
                prev_direction = direction
        
        print(f"Total brightness values: {len(brightness_history)}")
        print(f"Min brightness: {min(brightness_history):.1f}")
        print(f"Max brightness: {max(brightness_history):.1f}")
        print(f"Brightness range: {max(brightness_history) - min(brightness_history):.1f}")
        print(f"Significant changes (>5): {oscillations}")
        print(f"Direction changes: {direction_changes}")
        
        # Determine if flashing
        if direction_changes > total_frames / 10:  # More than 10% direction changes
            print("\n⚠️ FLASHING DETECTED: Too many brightness oscillations!")
        elif max(brightness_history) - min(brightness_history) < 20:
            print("\n⚠️ WARNING: Very small brightness range - text might not be visible!")
        else:
            print("\n✅ No significant flashing detected - brightness increases smoothly")
    
    # Plot brightness graph
    print("\nBrightness over time (simplified graph):")
    print("-" * 50)
    
    # Sample every 5 frames for graph
    sampled = brightness_history[::5]
    min_val = min(sampled)
    max_val = max(sampled)
    range_val = max_val - min_val
    
    if range_val > 0:
        for i, val in enumerate(sampled):
            normalized = int(((val - min_val) / range_val) * 40)
            bar = "#" * normalized
            print(f"F{i*5:3d}: {bar} ({val:.1f})")

if __name__ == "__main__":
    analyze_video_for_flashing("outputs/fade_debug_3sec_h264.mp4")