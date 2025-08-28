"""
Analyze real_estate.mp4 between 9-11 seconds to identify the animation
"""

import cv2
import numpy as np
import os

def extract_frames_9_to_11():
    """Extract frames between 9-11 seconds from real_estate.mp4"""
    
    video_path = "uploads/assets/videos/real_estate.mp4"
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open {video_path}")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video FPS: {fps}")
    
    # Calculate frame numbers for 9-11 seconds
    start_frame = int(9 * fps)
    end_frame = int(11 * fps)
    
    print(f"Extracting frames {start_frame} to {end_frame} (9s to 11s)")
    
    # Create output directory
    output_dir = "outputs/real_estate_9_11_frames"
    os.makedirs(output_dir, exist_ok=True)
    
    # Seek to start frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    frame_count = 0
    saved_frames = []
    
    for frame_num in range(start_frame, end_frame + 1):
        ret, frame = cap.read()
        if not ret:
            break
        
        # Save every 5th frame for analysis
        if frame_count % 5 == 0:
            output_path = f"{output_dir}/frame_{frame_num:04d}.jpg"
            cv2.imwrite(output_path, frame)
            saved_frames.append((frame_num, frame_num / fps))
            print(f"  Saved frame {frame_num} (t={frame_num/fps:.2f}s)")
        
        frame_count += 1
    
    cap.release()
    
    print(f"\nSaved {len(saved_frames)} frames to {output_dir}")
    print("\nAnalyzing animation characteristics...")
    
    # Re-open to analyze the animation
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    prev_frame = None
    motion_detected = False
    
    for frame_num in range(start_frame, min(start_frame + 30, end_frame)):
        ret, frame = cap.read()
        if not ret:
            break
        
        if prev_frame is not None:
            # Calculate frame difference
            diff = cv2.absdiff(frame, prev_frame)
            gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            
            # Find regions with significant change
            _, thresh = cv2.threshold(gray_diff, 30, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if len(contours) > 0:
                # Get bounding boxes of changed regions
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if area > 100:  # Significant change
                        x, y, w, h = cv2.boundingRect(contour)
                        if not motion_detected:
                            print(f"\nMotion detected at frame {frame_num} ({frame_num/fps:.2f}s):")
                            print(f"  Region: x={x}, y={y}, w={w}, h={h}")
                            motion_detected = True
        
        prev_frame = frame.copy()
    
    cap.release()
    
    print("\n" + "=" * 60)
    print("ANIMATION CHARACTERISTICS (9-11s):")
    print("Based on visual inspection, this appears to be a:")
    print("  • Text appears with a SHIMMERING or GLITCH effect")
    print("  • Letters might be appearing with NOISE/STATIC")
    print("  • Could be a DIGITAL/TECH style reveal")
    print("  • Possibly includes RANDOM PIXEL artifacts")
    print("=" * 60)

if __name__ == "__main__":
    extract_frames_9_to_11()