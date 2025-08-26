#!/usr/bin/env python3
"""
Visualize what the motion tracking is actually tracking in the video.
Shows the tracking box and tracked points in red.
"""

import cv2
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))
from base import track_object_in_video


def visualize_tracking(input_video: Path, output_path: Path, track_point=None):
    """
    Visualize object tracking with red boxes and tracking points.
    """
    # Open video
    cap = cv2.VideoCapture(str(input_video))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Default track point to center if not specified
    if track_point is None:
        track_point = (width // 2, height // 2)
    
    print(f"Video dimensions: {width}x{height}")
    print(f"Initial tracking point: {track_point}")
    
    # Define initial bounding box around track point
    bbox_size = 60
    initial_bbox = (
        max(0, track_point[0] - bbox_size // 2),
        max(0, track_point[1] - bbox_size // 2),
        bbox_size,
        bbox_size
    )
    
    print(f"Initial bounding box: {initial_bbox}")
    print("Tracking object through video...")
    
    # Track object through video
    cap.release()
    bboxes = track_object_in_video(Path(input_video), initial_bbox, use_replicate=False)
    
    print(f"Got {len(bboxes)} tracking boxes")
    
    # Reopen video for processing
    cap = cv2.VideoCapture(str(input_video))
    
    # Setup output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    # Add text explaining what we're showing
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx < len(bboxes):
            bbox = bboxes[frame_idx]
            x, y, w, h = bbox
            
            # Draw tracking box in RED
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 3)
            
            # Draw center point
            center_x = x + w // 2
            center_y = y + h // 2
            cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)
            
            # Draw crosshair
            cv2.line(frame, (center_x - 15, center_y), (center_x + 15, center_y), (0, 0, 255), 2)
            cv2.line(frame, (center_x, center_y - 15), (center_x, center_y + 15), (0, 0, 255), 2)
            
            # Add labels
            cv2.putText(frame, "TRACKED REGION", (x, y - 10), 
                       font, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
            
            # Show tracking info
            info_text = f"Frame {frame_idx}/{frame_count} | Box: ({x},{y}) {w}x{h}"
            cv2.putText(frame, info_text, (10, 30), 
                       font, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            
            # Add background for better visibility
            cv2.rectangle(frame, (5, 5), (400, 35), (0, 0, 0), -1)
            cv2.putText(frame, info_text, (10, 30), 
                       font, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            
            # Draw motion trail (last 10 positions)
            if frame_idx > 0:
                trail_length = min(10, frame_idx)
                for i in range(max(0, frame_idx - trail_length), frame_idx):
                    prev_bbox = bboxes[i]
                    prev_center = (prev_bbox[0] + prev_bbox[2] // 2, 
                                 prev_bbox[1] + prev_bbox[3] // 2)
                    alpha = (i - (frame_idx - trail_length)) / trail_length
                    color = (0, 0, int(255 * alpha))
                    cv2.circle(frame, prev_center, 3, color, -1)
        
        out.write(frame)
        frame_idx += 1
        
        if frame_idx % 30 == 0:
            print(f"Processed {frame_idx}/{frame_count} frames")
    
    cap.release()
    out.release()
    
    print(f"\nVisualization saved to: {output_path}")
    return output_path


def analyze_tracking_in_doremi():
    """
    Analyze what gets tracked in the do_re_mi video at different points.
    """
    input_video = Path("do_re_mi_quick_test/short_clip.mp4")
    
    if not input_video.exists():
        # Try the original
        input_video = Path("/Users/amirshachar/Desktop/Amir/Projects/Personal/toontune/backend/uploads/assets/videos/do_re_mi.mov")
        if not input_video.exists():
            print("Video not found!")
            return
    
    print("="*60)
    print("MOTION TRACKING VISUALIZATION")
    print("="*60)
    print(f"\nAnalyzing: {input_video}")
    
    # Create multiple tracking visualizations at different points
    output_dir = Path("tracking_visualization")
    output_dir.mkdir(exist_ok=True)
    
    # Get video dimensions
    cap = cv2.VideoCapture(str(input_video))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Extract first frame for analysis
    ret, first_frame = cap.read()
    cap.release()
    
    if ret:
        # Save first frame with grid overlay
        grid_frame = first_frame.copy()
        
        # Draw grid to show tracking regions
        grid_size = 60  # Same as bbox_size
        for x in range(0, width, grid_size):
            cv2.line(grid_frame, (x, 0), (x, height), (200, 200, 200), 1)
        for y in range(0, height, grid_size):
            cv2.line(grid_frame, (0, y), (width, y), (200, 200, 200), 1)
        
        # Mark different tracking points
        test_points = [
            ((width // 2, height // 2), "CENTER", (0, 0, 255)),
            ((width // 3, height // 3), "TOP-LEFT", (255, 0, 0)),
            ((2 * width // 3, height // 3), "TOP-RIGHT", (0, 255, 0)),
            ((width // 2, 2 * height // 3), "BOTTOM", (255, 255, 0)),
        ]
        
        for point, label, color in test_points:
            x, y = point
            # Draw tracking box
            cv2.rectangle(grid_frame, 
                         (x - grid_size // 2, y - grid_size // 2),
                         (x + grid_size // 2, y + grid_size // 2),
                         color, 2)
            cv2.circle(grid_frame, point, 5, color, -1)
            cv2.putText(grid_frame, label, (x + 10, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Save grid frame
        grid_path = output_dir / "tracking_grid_overlay.png"
        cv2.imwrite(str(grid_path), grid_frame)
        print(f"\nSaved tracking grid: {grid_path}")
    
    # Test different tracking points
    tracking_tests = [
        ("center", None),  # Default center tracking
        ("custom", (width // 3, height // 2)),  # Custom point
    ]
    
    for test_name, track_point in tracking_tests:
        print(f"\n--- Testing {test_name} tracking ---")
        output_path = output_dir / f"tracking_{test_name}.mp4"
        
        # Create visualization
        visualize_tracking(input_video, output_path, track_point)
    
    # Create a comparison showing what pixels are being tracked
    print("\n--- Creating pixel-level visualization ---")
    create_pixel_visualization(input_video, output_dir)
    
    print("\n" + "="*60)
    print("VISUALIZATION COMPLETE")
    print("="*60)
    print(f"\nOutputs in: {output_dir}/")
    print("- tracking_grid_overlay.png: Shows potential tracking regions")
    print("- tracking_center.mp4: Center point tracking")
    print("- tracking_custom.mp4: Custom point tracking")
    print("- tracking_pixels.png: Pixel-level tracking visualization")


def create_pixel_visualization(input_video: Path, output_dir: Path):
    """
    Create a static image showing which pixels are tracked over time.
    """
    cap = cv2.VideoCapture(str(input_video))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Track from center
    track_point = (width // 2, height // 2)
    bbox_size = 60
    initial_bbox = (
        max(0, track_point[0] - bbox_size // 2),
        max(0, track_point[1] - bbox_size // 2),
        bbox_size,
        bbox_size
    )
    
    # Get tracking data
    bboxes = track_object_in_video(input_video, initial_bbox, use_replicate=False)
    
    # Create heat map of tracked regions
    heat_map = np.zeros((height, width), dtype=np.float32)
    
    for bbox in bboxes:
        x, y, w, h = bbox
        # Add to heat map
        heat_map[y:y+h, x:x+w] += 1
    
    # Normalize heat map
    if heat_map.max() > 0:
        heat_map = (heat_map / heat_map.max() * 255).astype(np.uint8)
    
    # Apply colormap
    heat_map_colored = cv2.applyColorMap(heat_map, cv2.COLORMAP_JET)
    
    # Get first and last frame for reference
    ret, first_frame = cap.read()
    cap.set(cv2.CAP_PROP_POS_FRAMES, cap.get(cv2.CAP_PROP_FRAME_COUNT) - 1)
    ret, last_frame = cap.read()
    cap.release()
    
    # Create composite image
    if ret:
        # Blend heat map with first frame
        composite = cv2.addWeighted(first_frame, 0.5, heat_map_colored, 0.5, 0)
        
        # Create side-by-side comparison
        comparison = np.hstack([
            first_frame,
            heat_map_colored,
            composite
        ])
        
        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(comparison, "Original Frame", (10, 30), font, 1, (255, 255, 255), 2)
        cv2.putText(comparison, "Tracked Pixels (Heat Map)", (width + 10, 30), font, 1, (255, 255, 255), 2)
        cv2.putText(comparison, "Overlay", (2 * width + 10, 30), font, 1, (255, 255, 255), 2)
        
        output_path = output_dir / "tracking_pixels.png"
        cv2.imwrite(str(output_path), comparison)
        print(f"Saved pixel visualization: {output_path}")
        
        # Also save individual tracking box from first bbox
        if bboxes:
            x, y, w, h = bboxes[0]
            tracked_region = first_frame[y:y+h, x:x+w].copy()
            
            # Highlight edges in red
            cv2.rectangle(tracked_region, (0, 0), (w-1, h-1), (0, 0, 255), 2)
            
            # Scale up for better visibility
            scale = 4
            tracked_region_large = cv2.resize(tracked_region, (w * scale, h * scale), 
                                             interpolation=cv2.INTER_NEAREST)
            
            detail_path = output_dir / "tracked_region_detail.png"
            cv2.imwrite(str(detail_path), tracked_region_large)
            print(f"Saved tracked region detail: {detail_path}")


if __name__ == "__main__":
    analyze_tracking_in_doremi()