#!/usr/bin/env python3
"""
Test script for CoTracker3 pipeline
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.tracking.cotracker3 import CoTracker3
import time

def run_pipeline():
    """Run the complete CoTracker3 test pipeline"""
    
    print("="*60)
    print("🚀 COTRACKER3 TEST PIPELINE")
    print("="*60)
    
    # Initialize tracker
    print("\n📦 Initializing CoTracker3...")
    tracker = CoTracker3(model_type="cotracker3_online")
    
    # Input video - using downsampled version for efficiency
    input_video = "uploads/assets/videos/do_re_mi/scenes/edited/downsampled/scene_001.mp4"
    
    # Step 1: Find stable background segment
    print("\n🔍 Step 1: Finding stable background segment...")
    start_time, end_time, stability = tracker.find_stable_background_segment(
        input_video,
        segment_duration=5.0,
        sample_rate=10  # Sample every 10 frames for speed
    )
    
    # Step 2: Extract the segment
    print("\n✂️ Step 2: Extracting 5-second segment...")
    output_segment = "tests/tracking_test.mov"
    tracker.extract_video_segment(
        input_video,
        output_segment,
        start_time,
        duration=5.0
    )
    
    # Step 3: Find a background edge point
    print("\n🎯 Step 3: Finding background edge point...")
    edge_x, edge_y = tracker.find_background_edge_point(output_segment, frame_idx=0)
    
    # Step 4: Load video for tracking
    print("\n📹 Step 4: Loading video for tracking...")
    video = tracker.load_video(output_segment)
    
    # Step 5: Track the edge point
    print("\n🔍 Step 5: Tracking edge point through video...")
    points = [(0, edge_x, edge_y)]  # Track from first frame
    
    # Time the tracking operation
    track_start = time.time()
    tracks, visibility, tracking_time = tracker.track_points(video, points=points)
    
    # Step 6: Create visualization
    print("\n🎨 Step 6: Creating visualization...")
    output_video = "tests/tracking_test_tracked.mp4"
    tracker.visualize_tracks(
        video, 
        tracks, 
        visibility, 
        output_video,
        point_size=8,
        trail_length=15
    )
    
    # Results summary
    print("\n" + "="*60)
    print("📊 FINAL RESULTS")
    print("="*60)
    print(f"✅ Stable segment found: {start_time:.2f}s - {end_time:.2f}s")
    print(f"✅ Stability score: {stability:.2f} (lower is better)")
    print(f"✅ Edge point tracked: ({edge_x:.1f}, {edge_y:.1f})")
    print(f"✅ Tracking completed in: {tracking_time:.2f} seconds")
    print(f"✅ Processing speed: {video.shape[1] / tracking_time:.1f} FPS")
    print(f"✅ Input segment: tests/tracking_test.mov")
    print(f"✅ Output video: tests/tracking_test_tracked.mp4")
    print("="*60)
    
    return tracking_time


if __name__ == "__main__":
    try:
        elapsed = run_pipeline()
        print(f"\n🎉 Pipeline completed successfully!")
        print(f"   Total tracking time: {elapsed:.2f} seconds")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()