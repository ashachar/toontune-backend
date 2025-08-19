#!/usr/bin/env python3
"""
Display tracking results as individual frames
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def extract_key_frames():
    """Extract and display key frames from the tracking video"""
    
    video_path = "tests/tracking_test_tracked.mp4"
    
    if not Path(video_path).exists():
        print(f"❌ Video not found: {video_path}")
        return
    
    print(f"📹 Extracting frames from: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"  Total frames: {total_frames}")
    print(f"  FPS: {fps}")
    
    # Extract frames at regular intervals
    frame_indices = [0, total_frames//4, total_frames//2, 3*total_frames//4, total_frames-1]
    frames = []
    
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append((idx, frame_rgb))
    
    cap.release()
    
    # Create a figure showing the frames
    fig, axes = plt.subplots(1, len(frames), figsize=(20, 4))
    fig.suptitle('CoTracker3 Point Tracking Results - Background Edge Point Movement', fontsize=16)
    
    for i, (frame_idx, frame) in enumerate(frames):
        axes[i].imshow(frame)
        axes[i].set_title(f'Frame {frame_idx}\n(t={frame_idx/fps:.2f}s)')
        axes[i].axis('off')
    
    plt.tight_layout()
    
    # Save the figure
    output_path = "tests/tracking_results_frames.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Saved frame visualization to: {output_path}")
    
    # plt.show()  # Commented out for non-interactive mode
    
    return frames

def create_gif():
    """Create a GIF from the tracked video for easy viewing"""
    import imageio
    
    video_path = "tests/tracking_test_tracked.mp4"
    output_gif = "tests/tracking_results.gif"
    
    print(f"🎬 Creating GIF from tracked video...")
    
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    # Sample every 10th frame for smaller GIF
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % 10 == 0:
            # Convert BGR to RGB and resize
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Resize to make GIF smaller
            frame_small = cv2.resize(frame_rgb, (512, 232))
            frames.append(frame_small)
        
        frame_count += 1
    
    cap.release()
    
    if frames:
        imageio.mimsave(output_gif, frames, fps=10, loop=0)
        print(f"✅ Created GIF with {len(frames)} frames")
        print(f"✅ Saved to: {output_gif}")
        return output_gif
    
    return None

def analyze_point_movement():
    """Analyze the tracked point's movement pattern"""
    
    video_path = "tests/tracking_test_tracked.mp4"
    
    print(f"📊 Analyzing point movement in tracked video...")
    
    cap = cv2.VideoCapture(video_path)
    
    # Read first frame to identify the tracked point
    ret, first_frame = cap.read()
    if not ret:
        print("❌ Could not read video")
        return
    
    # Reset to beginning
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    # Track the brightest/most prominent point (the tracked marker)
    positions = []
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Find bright spots (tracked points are usually marked brightly)
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Get the largest bright spot
            largest = max(contours, key=cv2.contourArea)
            M = cv2.moments(largest)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                positions.append((frame_count, cx, cy))
        
        frame_count += 1
    
    cap.release()
    
    if positions:
        # Plot movement pattern
        frames, xs, ys = zip(*positions)
        
        plt.figure(figsize=(12, 6))
        
        # Plot 1: X-Y trajectory
        plt.subplot(1, 2, 1)
        plt.plot(xs, ys, 'b-', alpha=0.6, linewidth=2)
        plt.scatter(xs[0], ys[0], c='green', s=100, label='Start', zorder=5)
        plt.scatter(xs[-1], ys[-1], c='red', s=100, label='End', zorder=5)
        plt.xlabel('X Position (pixels)')
        plt.ylabel('Y Position (pixels)')
        plt.title('Tracked Point Trajectory')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.gca().invert_yaxis()  # Invert Y axis to match image coordinates
        
        # Plot 2: Movement over time
        plt.subplot(1, 2, 2)
        plt.plot(frames, xs, 'b-', label='X position', alpha=0.7)
        plt.plot(frames, ys, 'r-', label='Y position', alpha=0.7)
        plt.xlabel('Frame Number')
        plt.ylabel('Position (pixels)')
        plt.title('Point Movement Over Time')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.suptitle('CoTracker3 Point Movement Analysis', fontsize=14)
        plt.tight_layout()
        
        # Save analysis
        analysis_path = "tests/tracking_movement_analysis.png"
        plt.savefig(analysis_path, dpi=150, bbox_inches='tight')
        print(f"✅ Saved movement analysis to: {analysis_path}")
        
        # plt.show()  # Commented out for non-interactive mode
        
        # Print statistics
        print(f"\n📊 Movement Statistics:")
        print(f"  Total frames: {len(positions)}")
        print(f"  Start position: ({xs[0]}, {ys[0]})")
        print(f"  End position: ({xs[-1]}, {ys[-1]})")
        print(f"  Total X movement: {abs(xs[-1] - xs[0]):.1f} pixels")
        print(f"  Total Y movement: {abs(ys[-1] - ys[0]):.1f} pixels")
        print(f"  Max X: {max(xs)}, Min X: {min(xs)}")
        print(f"  Max Y: {max(ys)}, Min Y: {min(ys)}")

if __name__ == "__main__":
    print("🎯 CoTracker3 Results Viewer")
    print("="*50)
    
    # Extract and show key frames
    frames = extract_key_frames()
    
    # Create GIF for easy viewing
    gif_path = create_gif()
    
    # Analyze movement
    analyze_point_movement()
    
    print("\n✅ All visualizations complete!")
    print(f"  📸 Frame montage: tests/tracking_results_frames.png")
    print(f"  🎬 Animated GIF: tests/tracking_results.gif")
    print(f"  📊 Movement analysis: tests/tracking_movement_analysis.png")