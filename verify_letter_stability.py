#!/usr/bin/env python3
"""
Verify that letter positions remain stable when dissolve starts.
Focus on the transition at frame 90 where the issue occurred.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

def extract_frames(video_path, frame_indices):
    """Extract specific frames from video."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
    
    cap.release()
    return frames

def analyze_letter_stability(frames, frame_indices):
    """Analyze letter position stability across frames."""
    
    # Focus on the right side of the image where the last letter is
    h, w = frames[0].shape[:2]
    
    # Region for last letter (rightmost part of text)
    last_letter_x1 = int(w * 0.52)  # Right side of center
    last_letter_x2 = int(w * 0.65)
    text_y1 = int(h * 0.40)
    text_y2 = int(h * 0.50)
    
    print("\nLast Letter Position Analysis:")
    print("-" * 50)
    
    for i, (frame, idx) in enumerate(zip(frames, frame_indices)):
        # Extract the region where the last letter should be
        region = frame[text_y1:text_y2, last_letter_x1:last_letter_x2]
        
        # Calculate center of mass of bright pixels (text)
        gray = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY)
        bright_pixels = gray > 100
        
        if np.any(bright_pixels):
            y_coords, x_coords = np.where(bright_pixels)
            center_x = np.mean(x_coords) + last_letter_x1
            center_y = np.mean(y_coords) + text_y1
            
            print(f"Frame {idx}: Last letter center at ({center_x:.1f}, {center_y:.1f})")
            
            if i > 0:
                prev_frame = frames[i-1]
                prev_region = prev_frame[text_y1:text_y2, last_letter_x1:last_letter_x2]
                prev_gray = cv2.cvtColor(prev_region, cv2.COLOR_RGB2GRAY)
                prev_bright = prev_gray > 100
                
                if np.any(prev_bright):
                    prev_y, prev_x = np.where(prev_bright)
                    prev_center_x = np.mean(prev_x) + last_letter_x1
                    
                    x_shift = center_x - prev_center_x
                    print(f"  → X shift from frame {frame_indices[i-1]}: {x_shift:+.1f} pixels")
                    
                    if abs(x_shift) > 2:
                        print(f"  ⚠️  POSITION JUMP DETECTED!")
        else:
            print(f"Frame {idx}: No text detected in last letter region")

def main():
    # Critical frames around the dissolve transition
    # Note: Output video is at 30fps, so frame numbers are halved
    # Original frame 90 (dissolve start) = Output frame 45
    critical_frames = [43, 44, 45, 46, 47]  # Frames 86-94 in original
    
    print("Checking letter position stability at dissolve transition...")
    
    # Extract frames
    frames = extract_frames("start_animation_refactored.mp4", critical_frames)
    
    if not frames:
        print("❌ No frames could be extracted. Check if video exists and is valid.")
        return
    
    print(f"✓ Extracted {len(frames)} frames")
    
    # Analyze stability
    analyze_letter_stability(frames, critical_frames)
    
    # Create visual comparison
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    
    for i, (frame, idx) in enumerate(zip(frames, critical_frames)):
        axes[i].imshow(frame)
        
        if idx < 45:
            title = f"Frame {idx}\n(Before Dissolve)"
        elif idx == 45:
            title = f"Frame {idx}\n(Dissolve Starts)"
        else:
            title = f"Frame {idx}\n(During Dissolve)"
        
        axes[i].set_title(title, fontweight='bold')
        axes[i].axis('off')
        
        # Mark the region we're analyzing for the last letter
        h, w = frame.shape[:2]
        rect_x1 = int(w * 0.52)
        rect_x2 = int(w * 0.65)
        rect_y1 = int(h * 0.40)
        rect_y2 = int(h * 0.50)
        
        axes[i].add_patch(plt.Rectangle(
            (rect_x1, rect_y1), 
            rect_x2 - rect_x1, 
            rect_y2 - rect_y1,
            fill=False, 
            edgecolor='red', 
            linewidth=1,
            alpha=0.5
        ))
    
    plt.suptitle("Letter Position Stability Check - Transition to Dissolve", fontsize=14)
    plt.tight_layout()
    plt.savefig("letter_stability_check.png", dpi=150, bbox_inches='tight')
    print("\n✓ Saved visual analysis to letter_stability_check.png")
    
    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY:")
    if len(frames) >= 3:
        h, w = frames[0].shape[:2]
        # Check if there's a jump at frame 45 (dissolve start)
        frame_44_region = frames[1][int(h*0.40):int(h*0.50), int(w*0.52):int(w*0.65)]
        frame_45_region = frames[2][int(h*0.40):int(h*0.50), int(w*0.52):int(w*0.65)]
        
        diff = np.mean(np.abs(frame_45_region.astype(float) - frame_44_region.astype(float)))
        
        if diff < 5:
            print("✅ Letter positions appear STABLE at dissolve transition")
        else:
            print(f"⚠️  Potential position shift detected (diff={diff:.1f})")

if __name__ == "__main__":
    main()