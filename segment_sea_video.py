#!/usr/bin/env python3
"""
Segment sea_small.mov with multiple object tracking using SAM2
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.utils.sam2_api.video_segmentation import SAM2VideoSegmenter, ClickPoint, SegmentationConfig


def segment_sea_objects():
    """Segment multiple objects in the sea video with colored tracks"""
    
    # Check for API token
    api_token = os.environ.get("REPLICATE_API_TOKEN")
    if not api_token:
        print("=" * 60)
        print("⚠️  REPLICATE_API_TOKEN not set!")
        print("=" * 60)
        print("\nTo run SAM2 segmentation, you need to set your Replicate API token:")
        print("\n1. Get your token at: https://replicate.com/account/api-tokens")
        print("2. Set it in your terminal:")
        print("   export REPLICATE_API_TOKEN='your_token_here'")
        print("\n3. Then run this script again:")
        print("   python segment_sea_video.py")
        print("=" * 60)
        
        # For testing, let's set a temporary token if you provide it
        token = input("\n🔑 Paste your Replicate API token here (or press Enter to exit): ").strip()
        if token:
            os.environ["REPLICATE_API_TOKEN"] = token
            print("✅ Token set for this session!")
        else:
            print("Exiting. Please set your token and try again.")
            return
    
    # Video paths
    video_path = "uploads/assets/videos/sea_small.mov"
    output_path = "uploads/assets/videos/sea_small_multi_tracked.mp4"
    
    print("\n🎥 SAM2 Multi-Object Tracking")
    print(f"Input: {video_path}")
    print(f"Output: {output_path}")
    
    # Initialize segmenter
    segmenter = SAM2VideoSegmenter()
    
    # Define click points for multiple objects
    # These points will create separate colored tracks for different objects
    # Adjust these coordinates based on what you see in your video
    click_points = [
        # Object 1 (e.g., a fish) - tracked across frames
        ClickPoint(x=150, y=100, frame=0, label=1, object_id="fish_1"),
        ClickPoint(x=160, y=110, frame=5, label=1, object_id="fish_1"),
        ClickPoint(x=170, y=120, frame=10, label=1, object_id="fish_1"),
        ClickPoint(x=180, y=130, frame=15, label=1, object_id="fish_1"),
        
        # Object 2 (e.g., another fish) - different color
        ClickPoint(x=300, y=150, frame=0, label=1, object_id="fish_2"),
        ClickPoint(x=310, y=160, frame=5, label=1, object_id="fish_2"),
        ClickPoint(x=320, y=170, frame=10, label=1, object_id="fish_2"),
        ClickPoint(x=330, y=180, frame=15, label=1, object_id="fish_2"),
        
        # Object 3 (e.g., coral or rock) - stationary object
        ClickPoint(x=200, y=250, frame=0, label=1, object_id="coral_1"),
        ClickPoint(x=200, y=250, frame=10, label=1, object_id="coral_1"),
        ClickPoint(x=200, y=250, frame=20, label=1, object_id="coral_1"),
        
        # Object 4 (e.g., another moving element)
        ClickPoint(x=400, y=100, frame=0, label=1, object_id="element_1"),
        ClickPoint(x=380, y=110, frame=5, label=1, object_id="element_1"),
        ClickPoint(x=360, y=120, frame=10, label=1, object_id="element_1"),
        ClickPoint(x=340, y=130, frame=15, label=1, object_id="element_1"),
        
        # Object 5 (e.g., background element)
        ClickPoint(x=100, y=200, frame=0, label=1, object_id="element_2"),
        ClickPoint(x=110, y=205, frame=8, label=1, object_id="element_2"),
        ClickPoint(x=120, y=210, frame=16, label=1, object_id="element_2"),
    ]
    
    # Configuration for colored overlay output
    config = SegmentationConfig(
        mask_type="highlighted",     # This will show different colors for each object
        annotation_type="mask",       # Show masks (not bounding boxes)
        output_video=True,           # Output as video file
        video_fps=30,                # Original frame rate
        output_frame_interval=1      # Process every frame for smooth tracking
    )
    
    # Count unique objects
    unique_objects = set(p.object_id for p in click_points)
    print(f"\n📍 Tracking {len(unique_objects)} separate objects:")
    for obj in unique_objects:
        points_count = len([p for p in click_points if p.object_id == obj])
        print(f"   • {obj}: {points_count} tracking points")
    
    print("\n🚀 Sending to Replicate for processing...")
    print("This may take 1-2 minutes depending on video length and complexity.")
    
    try:
        # Run segmentation
        result = segmenter.segment_video_advanced(
            video_path=video_path,
            click_points=click_points,
            config=config,
            output_path=output_path
        )
        
        print(f"\n✅ Success! Multi-object tracking complete!")
        print(f"📹 Segmented video saved to: {result}")
        print("\n🎨 Each tracked object appears in a different color:")
        print("   • fish_1: Color 1")
        print("   • fish_2: Color 2")
        print("   • coral_1: Color 3")
        print("   • element_1: Color 4")
        print("   • element_2: Color 5")
        
        # Open the video on macOS
        if sys.platform == "darwin" and Path(output_path).exists():
            print("\n🖥️  Opening video...")
            os.system(f"open '{output_path}'")
        
        return result
            
    except Exception as e:
        print(f"\n❌ Error during segmentation: {e}")
        print("\nTroubleshooting:")
        print("1. Check your internet connection")
        print("2. Verify your Replicate API token is valid")
        print("3. Ensure you have sufficient credits on Replicate")
        raise


def segment_with_grid_points():
    """Alternative approach: Use a grid of points for automatic detection"""
    
    video_path = "uploads/assets/videos/sea_small.mov"
    output_path = "uploads/assets/videos/sea_small_grid_tracked.mp4"
    
    print("\n🎯 Using grid-based detection for automatic object discovery")
    
    segmenter = SAM2VideoSegmenter()
    
    # Create a grid of points to detect objects automatically
    click_points = []
    
    # Grid configuration
    grid_x = [100, 200, 300, 400, 500]  # X coordinates
    grid_y = [75, 150, 225, 300]        # Y coordinates
    frames = [0, 10, 20]                # Sample frames
    
    object_counter = 0
    for frame in frames:
        for y in grid_y:
            for x in grid_x:
                object_counter += 1
                click_points.append(
                    ClickPoint(
                        x=x,
                        y=y,
                        frame=frame,
                        label=1,
                        object_id=f"auto_object_{object_counter}"
                    )
                )
    
    config = SegmentationConfig(
        mask_type="highlighted",
        output_video=True,
        video_fps=30
    )
    
    print(f"📍 Created {len(click_points)} detection points")
    print(f"📹 Processing video...")
    
    try:
        result = segmenter.segment_video_advanced(
            video_path=video_path,
            click_points=click_points,
            config=config,
            output_path=output_path
        )
        
        print(f"\n✅ Grid-based segmentation complete!")
        print(f"📹 Output: {result}")
        
        if sys.platform == "darwin" and Path(output_path).exists():
            os.system(f"open '{output_path}'")
        
        return result
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Segment sea video with SAM2")
    parser.add_argument(
        "--grid",
        action="store_true",
        help="Use grid-based automatic detection instead of manual points"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🌊 SAM2 Sea Video Segmentation")
    print("=" * 60)
    
    if args.grid:
        segment_with_grid_points()
    else:
        segment_sea_objects()