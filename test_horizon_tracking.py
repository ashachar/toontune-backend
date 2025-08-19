#!/usr/bin/env python3
"""
Test horizon point tracking with CoTracker3
Uses DeepLabV3 for horizon detection with Hough transform fallback
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.tracking.cotracker3 import CoTracker3
from utils.lines_detector import HorizonDetector
import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
from pathlib import Path


def run_horizon_tracking_pipeline():
    """Run the complete horizon tracking pipeline"""
    
    print("=" * 60)
    print("🌅 HORIZON POINT TRACKING PIPELINE")
    print("=" * 60)
    
    # Initialize detectors
    print("\n📦 Initializing systems...")
    horizon_detector = HorizonDetector(use_fast_model=True)  # Use fast MobileNet
    tracker = CoTracker3(model_type="cotracker3_online")
    
    # Use the same test video we created earlier
    test_video = "tests/tracking_test.mov"
    
    if not Path(test_video).exists():
        print(f"❌ Test video not found: {test_video}")
        print("   Please run test_cotracker_pipeline.py first")
        return None
    
    # Step 1: Load first frame for horizon detection
    print("\n🖼️ Step 1: Loading first frame...")
    cap = cv2.VideoCapture(test_video)
    ret, first_frame = cap.read()
    cap.release()
    
    if not ret:
        print("❌ Could not read first frame")
        return None
    
    print(f"   Frame shape: {first_frame.shape}")
    
    # Step 2: Detect horizon
    print("\n🌅 Step 2: Detecting horizon line...")
    horizon_result = horizon_detector.detect(first_frame)
    
    if horizon_result['horizon_points'] is None:
        print("❌ No horizon detected!")
        return None
    
    print(f"   Method used: {horizon_result['method']}")
    print(f"   Detection time: {horizon_result['time_ms']:.1f}ms")
    print(f"   Horizon angle: {horizon_result['angle']:.1f}°")
    print(f"   Confidence: {horizon_result['confidence']:.1%}")
    
    # Step 3: Select tracking point on horizon
    print("\n🎯 Step 3: Selecting point on horizon...")
    
    # Try different positions on the horizon
    positions = ['center', 'left', 'right']
    selected_position = 'center'  # Default to center
    
    horizon_x, horizon_y = horizon_detector.select_tracking_point(
        horizon_result['horizon_points'],
        position=selected_position
    )
    
    print(f"   Selected {selected_position} point: ({horizon_x:.1f}, {horizon_y:.1f})")
    
    # Step 4: Visualize horizon detection
    print("\n🎨 Step 4: Creating horizon visualization...")
    vis_frame = horizon_detector.visualize_detection(
        first_frame,
        horizon_result,
        save_path="tests/horizon_detection.png"
    )
    
    # Mark the selected tracking point
    cv2.circle(vis_frame, (int(horizon_x), int(horizon_y)), 8, (255, 0, 255), -1)
    cv2.circle(vis_frame, (int(horizon_x), int(horizon_y)), 10, (255, 255, 255), 2)
    cv2.putText(vis_frame, "Track Point", 
               (int(horizon_x) + 15, int(horizon_y) - 10),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
    cv2.imwrite("tests/horizon_with_track_point.png", vis_frame)
    
    # Step 5: Load video and track the horizon point
    print("\n📹 Step 5: Loading video for tracking...")
    video = tracker.load_video(test_video)
    
    # Step 6: Track the horizon point
    print("\n🔍 Step 6: Tracking horizon point through video...")
    points = [(0, horizon_x, horizon_y)]  # Track from first frame
    
    # Enable MPS fallback for grid_sampler_3d
    import os
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
    
    # Time the tracking operation
    track_start = time.time()
    tracks, visibility, tracking_time = tracker.track_points(video, points=points)
    
    # Step 7: Create visualization
    print("\n🎨 Step 7: Creating tracking visualization...")
    output_video = "tests/horizon_tracking_result.mp4"
    tracker.visualize_tracks(
        video, 
        tracks, 
        visibility, 
        output_video,
        point_size=8,
        trail_length=20
    )
    
    # Step 8: Analyze movement
    print("\n📊 Step 8: Analyzing horizon point movement...")
    tracks_np = tracks[0].cpu().numpy()
    
    # Calculate movement statistics
    x_positions = tracks_np[:, 0, 0]
    y_positions = tracks_np[:, 0, 1]
    
    x_movement = np.max(x_positions) - np.min(x_positions)
    y_movement = np.max(y_positions) - np.min(y_positions)
    total_movement = np.sqrt(x_movement**2 + y_movement**2)
    
    # Results summary
    print("\n" + "=" * 60)
    print("📊 HORIZON TRACKING RESULTS")
    print("=" * 60)
    print(f"✅ Horizon Detection:")
    print(f"   Method: {horizon_result['method']}")
    print(f"   Angle: {horizon_result['angle']:.1f}°")
    print(f"   Detection time: {horizon_result['time_ms']:.1f}ms")
    print(f"\n✅ Point Tracking:")
    print(f"   Tracked point: ({horizon_x:.1f}, {horizon_y:.1f})")
    print(f"   Tracking time: {tracking_time:.2f} seconds")
    print(f"   Processing speed: {video.shape[1] / tracking_time:.1f} FPS")
    print(f"\n✅ Movement Analysis:")
    print(f"   X movement: {x_movement:.1f} pixels")
    print(f"   Y movement: {y_movement:.1f} pixels")
    print(f"   Total movement: {total_movement:.1f} pixels")
    print(f"   Start position: ({x_positions[0]:.1f}, {y_positions[0]:.1f})")
    print(f"   End position: ({x_positions[-1]:.1f}, {y_positions[-1]:.1f})")
    print(f"\n✅ Output Files:")
    print(f"   Horizon detection: tests/horizon_detection.png")
    print(f"   Track point: tests/horizon_with_track_point.png")
    print(f"   Tracked video: tests/horizon_tracking_result.mp4")
    print("=" * 60)
    
    return {
        'horizon_result': horizon_result,
        'tracking_time': tracking_time,
        'movement_stats': {
            'x_movement': x_movement,
            'y_movement': y_movement,
            'total_movement': total_movement
        }
    }


def create_comparison_visualization():
    """Create a comparison between edge tracking and horizon tracking"""
    
    print("\n📊 Creating comparison visualization...")
    
    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle('Edge Tracking vs Horizon Tracking Comparison', fontsize=16, fontweight='bold')
    
    # Load images
    edge_detection = cv2.imread("tests/tracking_test.mov")
    horizon_detection = cv2.imread("tests/horizon_detection.png")
    horizon_with_point = cv2.imread("tests/horizon_with_track_point.png")
    
    # Convert BGR to RGB for matplotlib
    if edge_detection is not None:
        edge_detection = cv2.cvtColor(edge_detection, cv2.COLOR_BGR2RGB)
    if horizon_detection is not None:
        horizon_detection = cv2.cvtColor(horizon_detection, cv2.COLOR_BGR2RGB)
    if horizon_with_point is not None:
        horizon_with_point = cv2.cvtColor(horizon_with_point, cv2.COLOR_BGR2RGB)
    
    # Row 1: Edge tracking (previous test)
    axes[0, 0].set_title('Edge Detection\n(Original Method)', fontsize=11)
    axes[0, 0].text(0.5, 0.5, 'Background Edge\nPoint: (199, 11)', 
                   ha='center', va='center', fontsize=10)
    axes[0, 0].set_xticks([])
    axes[0, 0].set_yticks([])
    
    axes[0, 1].set_title('Edge Tracking Result', fontsize=11)
    axes[0, 1].text(0.5, 0.5, 'Movement: <1 pixel\nTime: 13.82s', 
                   ha='center', va='center', fontsize=10)
    axes[0, 1].set_xticks([])
    axes[0, 1].set_yticks([])
    
    axes[0, 2].set_title('Edge Stability', fontsize=11)
    axes[0, 2].text(0.5, 0.5, 'Extremely Stable\n99.9% stationary', 
                   ha='center', va='center', fontsize=10,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))
    axes[0, 2].set_xticks([])
    axes[0, 2].set_yticks([])
    
    # Row 2: Horizon tracking (new test)
    if horizon_detection is not None:
        axes[1, 0].imshow(horizon_detection)
        axes[1, 0].set_title('Horizon Detection\n(DeepLabV3/Hough)', fontsize=11)
    else:
        axes[1, 0].text(0.5, 0.5, 'Horizon Detection\n(Pending)', 
                       ha='center', va='center', fontsize=10)
    axes[1, 0].set_xticks([])
    axes[1, 0].set_yticks([])
    
    if horizon_with_point is not None:
        axes[1, 1].imshow(horizon_with_point)
        axes[1, 1].set_title('Horizon Track Point', fontsize=11)
    else:
        axes[1, 1].text(0.5, 0.5, 'Horizon Tracking\n(Running...)', 
                       ha='center', va='center', fontsize=10)
    axes[1, 1].set_xticks([])
    axes[1, 1].set_yticks([])
    
    axes[1, 2].set_title('Horizon Movement', fontsize=11)
    axes[1, 2].text(0.5, 0.5, 'Analyzing...', 
                   ha='center', va='center', fontsize=10)
    axes[1, 2].set_xticks([])
    axes[1, 2].set_yticks([])
    
    # Add method labels
    axes[0, 0].text(-0.3, 0.5, 'EDGE\nMETHOD', rotation=0, fontsize=12, fontweight='bold',
                   ha='center', va='center', transform=axes[0, 0].transAxes)
    axes[1, 0].text(-0.3, 0.5, 'HORIZON\nMETHOD', rotation=0, fontsize=12, fontweight='bold',
                   ha='center', va='center', transform=axes[1, 0].transAxes)
    
    plt.tight_layout()
    
    # Save comparison
    output_path = "tests/tracking_method_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"✅ Comparison saved to: {output_path}")
    
    return output_path


if __name__ == "__main__":
    try:
        # Run horizon tracking pipeline
        results = run_horizon_tracking_pipeline()
        
        if results:
            # Create comparison visualization
            comparison_path = create_comparison_visualization()
            
            print(f"\n🎉 Horizon tracking pipeline completed successfully!")
            print(f"   Comparison visualization: {comparison_path}")
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()