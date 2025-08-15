#!/usr/bin/env python3
"""
Test script for SAM2 video segmentation on sea_small.mov
"""

import os
import sys
from pathlib import Path

# Add parent directory to path to import utils
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.utils.sam2_api.video_segmentation import SAM2VideoSegmenter, ClickPoint, SegmentationConfig


def segment_sea_video(video_path=None, output_path=None):
    """Segment objects in the sea_small.mov video"""
    
    # Input and output paths
    if video_path is None:
        video_path = "uploads/assets/videos/sea_small.mov"
    if output_path is None:
        output_path = video_path.replace(".mov", "_segmented.mp4").replace(".mp4", "_segmented.mp4")
    
    # Initialize the segmenter
    print("Initializing SAM2 Video Segmenter...")
    segmenter = SAM2VideoSegmenter()
    
    # Define click points for segmentation
    # You can adjust these coordinates based on what you want to segment in the video
    # These are example points - you may need to adjust based on your video content
    click_points = [
        ClickPoint(x=200, y=150, frame=0, label=1, object_id="fish_1"),
        ClickPoint(x=300, y=200, frame=0, label=1, object_id="fish_2"),
        ClickPoint(x=150, y=250, frame=10, label=1, object_id="coral_1"),
        ClickPoint(x=400, y=180, frame=20, label=1, object_id="fish_3"),
    ]
    
    # Configuration for output
    config = SegmentationConfig(
        mask_type="highlighted",  # Show colored overlay on detected objects
        annotation_type="mask",    # Show mask (can be "bbox" or "both" for bounding boxes)
        output_video=True,         # Output as video
        video_fps=30,              # Output frame rate
        output_frame_interval=1    # Process every frame
    )
    
    print(f"Processing video: {video_path}")
    print(f"Click points defined: {len(click_points)} objects")
    for point in click_points:
        print(f"  - {point.object_id}: ({point.x}, {point.y}) at frame {point.frame}")
    
    try:
        # Run segmentation
        result = segmenter.segment_video_advanced(
            video_path=video_path,
            click_points=click_points,
            config=config,
            output_path=output_path
        )
        
        print(f"\n✅ Segmentation complete!")
        print(f"Output saved to: {result}")
        
        # Display the video if running locally
        if Path(output_path).exists():
            print(f"\nTo view the segmented video, open: {output_path}")
            # Optionally open the video automatically (macOS)
            if sys.platform == "darwin":
                os.system(f"open '{output_path}'")
        
        return result
        
    except Exception as e:
        print(f"\n❌ Error during segmentation: {e}")
        raise


def segment_with_auto_points():
    """Alternative: Segment video with automatically detected points across frames"""
    
    video_path = "uploads/assets/videos/sea_small.mov"
    output_path = "uploads/assets/videos/sea_small_auto_segmented.mp4"
    
    print("Initializing SAM2 with automatic point detection...")
    segmenter = SAM2VideoSegmenter()
    
    # Simple approach: click on multiple points across different frames
    # This creates a grid of points to detect various objects
    click_points = []
    
    # Create a grid of points at different frames
    frame_intervals = [0, 15, 30, 45, 60]  # Sample frames
    grid_points = [
        (100, 100), (200, 100), (300, 100), (400, 100),
        (100, 200), (200, 200), (300, 200), (400, 200),
        (100, 300), (200, 300), (300, 300), (400, 300),
    ]
    
    # Add points from grid at different frames
    for frame_idx, frame in enumerate(frame_intervals[:3]):  # Use first 3 frames
        for point_idx, (x, y) in enumerate(grid_points[:4]):  # Use first 4 points
            click_points.append(
                ClickPoint(
                    x=x,
                    y=y,
                    frame=frame,
                    label=1,
                    object_id=f"object_{frame_idx}_{point_idx}"
                )
            )
    
    config = SegmentationConfig(
        mask_type="highlighted",
        output_video=True,
        video_fps=30
    )
    
    print(f"Using {len(click_points)} automatic detection points")
    
    try:
        result = segmenter.segment_video_advanced(
            video_path=video_path,
            click_points=click_points,
            config=config,
            output_path=output_path
        )
        
        print(f"\n✅ Auto-segmentation complete!")
        print(f"Output saved to: {result}")
        
        return result
        
    except Exception as e:
        print(f"\n❌ Error during auto-segmentation: {e}")
        raise


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test SAM2 video segmentation")
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Use automatic point detection instead of manual points"
    )
    parser.add_argument(
        "--video",
        type=str,
        help="Path to video file to segment (overrides default)"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Path for output video file"
    )
    
    args = parser.parse_args()
    
    # Check if Replicate API token is set
    if not os.environ.get("REPLICATE_API_TOKEN"):
        print("⚠️  REPLICATE_API_TOKEN not set!")
        print("Please set your Replicate API token:")
        print("  export REPLICATE_API_TOKEN='your_token_here'")
        print("\nGet your token at: https://replicate.com/account/api-tokens")
        sys.exit(1)
    
    # Override paths if provided
    if args.video:
        video_path = args.video
        if not Path(video_path).exists():
            print(f"❌ Video file not found: {video_path}")
            sys.exit(1)
    else:
        # Use a default video if sea_small.mov doesn't exist
        default_videos = [
            "uploads/assets/videos/sea_small.mov",
            "backend/test_output/composited_video.mp4",
            "app/hand_drawing_woman.mp4"
        ]
        video_path = None
        for video in default_videos:
            if Path(video).exists():
                video_path = video
                print(f"Using video: {video_path}")
                break
        
        if not video_path:
            print("❌ No video file found. Please specify a video with --video flag")
            print("Example: python test_sam2_video.py --video path/to/your/video.mp4")
            sys.exit(1)
    
    output_path = args.output if args.output else None
    
    # Run segmentation
    if args.auto:
        segment_with_auto_points()
    else:
        segment_sea_video(video_path, output_path)