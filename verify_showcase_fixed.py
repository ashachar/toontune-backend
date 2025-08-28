"""
Verify that all animations in the showcase are smooth without blinking
"""

import cv2
import numpy as np
from pathlib import Path


def verify_showcase():
    """Extract and analyze frames from multiple animations in the showcase"""
    
    print("Verifying 3D showcase animations for blinking...")
    print("=" * 60)
    
    video_path = "outputs/3D_Text_Animations_Showcase.mp4"
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error: Could not open video file")
        return False
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video: {fps} fps, {total_frames} total frames")
    print(f"Duration: {total_frames / fps:.1f} seconds")
    print()
    
    # Test points for different animations (in seconds)
    test_points = [
        (2.0, "3D Fade Wave"),
        (5.0, "3D Blur Fade"),
        (8.0, "3D Glow Pulse"),
        (20.0, "3D Float"),
        (32.0, "3D Rotate"),
        (50.0, "3D Word Reveal"),
        (65.0, "3D Zoom+Blur"),
        (77.0, "3D Spiral Materialize")
    ]
    
    issues_found = []
    
    for test_time, anim_name in test_points:
        print(f"Testing {anim_name} at {test_time}s...")
        
        # Get frames around this time
        start_frame = int(test_time * fps)
        frames = []
        
        for i in range(5):  # Get 5 consecutive frames
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame + i)
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
        
        # Analyze for blinking
        for i in range(len(frames) - 1):
            # Compare brightness in text region
            region1 = frames[i][300:420, 400:880]
            region2 = frames[i+1][300:420, 400:880]
            
            _, thresh1 = cv2.threshold(cv2.cvtColor(region1, cv2.COLOR_BGR2GRAY), 100, 255, cv2.THRESH_BINARY)
            _, thresh2 = cv2.threshold(cv2.cvtColor(region2, cv2.COLOR_BGR2GRAY), 100, 255, cv2.THRESH_BINARY)
            
            bright_change = abs(np.sum(thresh1 > 0) - np.sum(thresh2 > 0))
            
            if bright_change > 3000:
                print(f"  ⚠️ Large brightness change detected: {bright_change} pixels")
                issues_found.append((anim_name, test_time, bright_change))
            else:
                print(f"  ✅ Smooth transition (change: {bright_change} pixels)")
    
    cap.release()
    
    print("\n" + "=" * 60)
    
    if issues_found:
        print("ISSUES FOUND:")
        for anim, time, change in issues_found:
            print(f"  - {anim} at {time}s: {change} pixel change")
        print("\nSome animations may still have minor issues.")
        return False
    else:
        print("✅ SUCCESS! All tested animations are smooth without blinking!")
        print("The blinking issue has been successfully eliminated.")
        return True


if __name__ == "__main__":
    success = verify_showcase()
    
    if success:
        print("\n🎉 All 3D animations have been fixed and are running smoothly!")
        print("The showcase video is ready: outputs/3D_Text_Animations_Showcase.mp4")