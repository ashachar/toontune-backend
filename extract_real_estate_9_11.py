"""
Extract and analyze frames from real_estate.mov between 9-11 seconds
"""

import cv2
import numpy as np
import os

def extract_and_analyze():
    video_path = "uploads/assets/videos/real_estate.mov"
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open {video_path}")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video info: {fps:.2f} fps, {total_frames} total frames")
    
    # Calculate frame range for 9-11 seconds
    start_frame = int(9 * fps)
    end_frame = int(11 * fps)
    
    print(f"Extracting frames {start_frame} to {end_frame} (9s to 11s)")
    
    # Create output directory
    output_dir = "outputs/real_estate_9_11_analysis"
    os.makedirs(output_dir, exist_ok=True)
    
    # Seek to start frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    # Extract every 3rd frame for detailed analysis
    frame_data = []
    for frame_num in range(start_frame, end_frame + 1):
        ret, frame = cap.read()
        if not ret:
            break
        
        # Save key frames
        if (frame_num - start_frame) % 3 == 0:
            timestamp = frame_num / fps
            output_path = f"{output_dir}/frame_{frame_num:04d}_t{timestamp:.2f}.jpg"
            cv2.imwrite(output_path, frame)
            frame_data.append((frame_num, timestamp, frame))
            print(f"  Saved frame {frame_num} (t={timestamp:.2f}s)")
    
    cap.release()
    
    print(f"\nAnalyzing animation pattern...")
    
    # Analyze differences between consecutive frames
    if len(frame_data) > 1:
        for i in range(1, len(frame_data)):
            prev_frame = frame_data[i-1][2]
            curr_frame = frame_data[i][2]
            timestamp = frame_data[i][1]
            
            # Calculate difference
            diff = cv2.absdiff(curr_frame, prev_frame)
            gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            
            # Find significant changes
            _, thresh = cv2.threshold(gray_diff, 30, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if len(contours) > 0:
                # Find bounding box of all changes
                x_min, y_min = float('inf'), float('inf')
                x_max, y_max = 0, 0
                
                for contour in contours:
                    x, y, w, h = cv2.boundingRect(contour)
                    x_min = min(x_min, x)
                    y_min = min(y_min, y)
                    x_max = max(x_max, x + w)
                    y_max = max(y_max, y + h)
                
                if i == 1:
                    print(f"\nAnimation detected at t={timestamp:.2f}s:")
                    print(f"  Region: ({x_min}, {y_min}) to ({x_max}, {y_max})")
                    print(f"  Size: {x_max - x_min}x{y_max - y_min}")
    
    print("\nFrame extraction complete!")
    print(f"Frames saved to: {output_dir}")
    return output_dir

if __name__ == "__main__":
    output_dir = extract_and_analyze()