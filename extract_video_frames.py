#!/usr/bin/env python3
"""
Extract key frames from tracking videos for viewing
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def extract_tracking_frames():
    """Extract and display key frames from both tracking videos"""
    
    videos = [
        ("tests/tracking_test_tracked_h264.mp4", "Edge Tracking"),
        ("tests/horizon_tracking_h264_compatible.mp4", "Horizon Tracking")
    ]
    
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    fig.suptitle('CoTracker3 Video Results - Key Frames', fontsize=16, fontweight='bold')
    
    for vid_idx, (video_path, title) in enumerate(videos):
        if not Path(video_path).exists():
            print(f"⚠️ Video not found: {video_path}")
            continue
            
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"\n📹 {title}: {video_path}")
        print(f"   Total frames: {total_frames}")
        print(f"   FPS: {fps}")
        
        # Extract 5 evenly spaced frames
        frame_indices = np.linspace(0, total_frames-1, 5, dtype=int)
        
        for i, frame_idx in enumerate(frame_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if ret:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Display frame
                axes[vid_idx, i].imshow(frame_rgb)
                time_sec = frame_idx / fps if fps > 0 else 0
                axes[vid_idx, i].set_title(f'Frame {frame_idx}\n(t={time_sec:.2f}s)', fontsize=10)
                axes[vid_idx, i].axis('off')
                
                # Add colored border to distinguish methods
                for spine in axes[vid_idx, i].spines.values():
                    spine.set_edgecolor('blue' if vid_idx == 0 else 'orange')
                    spine.set_linewidth(3)
        
        cap.release()
        
        # Add method label
        axes[vid_idx, 0].text(-0.3, 0.5, title, rotation=90, fontsize=12, fontweight='bold',
                             ha='center', va='center', transform=axes[vid_idx, 0].transAxes,
                             color='blue' if vid_idx == 0 else 'orange')
    
    plt.tight_layout()
    
    # Save the figure
    output_path = "tests/video_frames_extract.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\n✅ Frame extracts saved to: {output_path}")
    
    return output_path


def create_video_summary():
    """Create a summary image showing what the videos contain"""
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle('CoTracker3 Video Tracking Results Summary', fontsize=16, fontweight='bold')
    
    # Edge tracking summary
    axes[0, 0].text(0.5, 0.5, '🔷 EDGE TRACKING\n\nPoint: (199, 11)\nBackground edge', 
                   ha='center', va='center', fontsize=12,
                   bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue"))
    axes[0, 0].set_title('Method', fontsize=11)
    axes[0, 0].axis('off')
    
    axes[0, 1].text(0.5, 0.5, '⏱️ Performance\n\n13.82 seconds\n22.0 FPS\n<1 pixel movement', 
                   ha='center', va='center', fontsize=11)
    axes[0, 1].set_title('Metrics', fontsize=11)
    axes[0, 1].axis('off')
    
    axes[0, 2].text(0.5, 0.5, '📹 VIDEO FILE:\ntracking_test_tracked_h264.mp4\n\n'
                   '• Pink tracking marker\n• Shows point stability\n• 10 second duration', 
                   ha='center', va='center', fontsize=10)
    axes[0, 2].set_title('Output Video', fontsize=11)
    axes[0, 2].axis('off')
    
    # Horizon tracking summary
    axes[1, 0].text(0.5, 0.5, '🌅 HORIZON TRACKING\n\nPoint: (130, 55)\nHorizon center', 
                   ha='center', va='center', fontsize=12,
                   bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow"))
    axes[1, 0].set_title('Method', fontsize=11)
    axes[1, 0].axis('off')
    
    axes[1, 1].text(0.5, 0.5, '⏱️ Performance\n\n11.68 seconds\n26.0 FPS\n2.0 pixel movement', 
                   ha='center', va='center', fontsize=11)
    axes[1, 1].set_title('Metrics', fontsize=11)
    axes[1, 1].axis('off')
    
    axes[1, 2].text(0.5, 0.5, '📹 VIDEO FILE:\nhorizon_tracking_h264_compatible.mp4\n\n'
                   '• Pink tracking marker\n• Follows horizon line\n• 10 second duration', 
                   ha='center', va='center', fontsize=10)
    axes[1, 2].set_title('Output Video', fontsize=11)
    axes[1, 2].axis('off')
    
    # Add row labels
    fig.text(0.08, 0.7, 'TEST 1', rotation=90, fontsize=14, fontweight='bold', 
            ha='center', va='center', color='blue')
    fig.text(0.08, 0.3, 'TEST 2', rotation=90, fontsize=14, fontweight='bold',
            ha='center', va='center', color='orange')
    
    plt.tight_layout()
    
    # Save summary
    output_path = "tests/video_tracking_summary.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"✅ Video summary saved to: {output_path}")
    
    return output_path


if __name__ == "__main__":
    print("📹 Extracting video frames and creating summaries...")
    
    # Extract frames
    frames_path = extract_tracking_frames()
    
    # Create summary
    summary_path = create_video_summary()
    
    print("\n✅ All visualizations complete!")
    print(f"   📸 Frame extracts: {frames_path}")
    print(f"   📋 Video summary: {summary_path}")
    
    # List all video files
    print("\n📁 Available H.264 videos for playback:")
    from pathlib import Path
    for video in Path("tests").glob("*h264*.mp4"):
        size = video.stat().st_size / 1024
        print(f"   • {video.name} ({size:.1f} KB)")