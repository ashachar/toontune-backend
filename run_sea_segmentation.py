#!/usr/bin/env python3
"""
Run SAM2 segmentation on sea_small.mov with multi-object tracking
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.utils.sam2_api.video_segmentation import SAM2VideoSegmenter, ClickPoint, SegmentationConfig

# Load environment variables from .env file
load_dotenv('./backend/.env')

def segment_sea_video():
    """Segment multiple objects in sea video with colored tracks"""
    
    # Get API key from environment (supports both KEY and TOKEN naming)
    api_key = os.environ.get("REPLICATE_API_KEY") or os.environ.get("REPLICATE_API_TOKEN")
    
    if api_key:
        # Set as REPLICATE_API_TOKEN which the library expects
        os.environ["REPLICATE_API_TOKEN"] = api_key
        print(f"✅ Using Replicate API key from .env file")
    else:
        print("❌ No Replicate API key found in environment")
        return
    
    # Video paths
    video_path = "uploads/assets/videos/sea_small.mov"
    output_path = "uploads/assets/videos/sea_small_segmented.mp4"
    
    print("\n🌊 SAM2 Multi-Object Video Segmentation")
    print(f"📹 Input: {video_path}")
    print(f"📹 Output: {output_path}")
    
    # Initialize segmenter
    segmenter = SAM2VideoSegmenter()
    
    # Define click points for multiple objects with different tracks
    # Each unique object_id will get a different color
    click_points = [
        # Object 1 - Track across multiple frames (e.g., fish swimming left to right)
        ClickPoint(x=150, y=120, frame=0, label=1, object_id="moving_object_1"),
        ClickPoint(x=180, y=125, frame=5, label=1, object_id="moving_object_1"),
        ClickPoint(x=210, y=130, frame=10, label=1, object_id="moving_object_1"),
        ClickPoint(x=240, y=135, frame=15, label=1, object_id="moving_object_1"),
        
        # Object 2 - Different track (e.g., another fish)
        ClickPoint(x=350, y=180, frame=0, label=1, object_id="moving_object_2"),
        ClickPoint(x=340, y=185, frame=5, label=1, object_id="moving_object_2"),
        ClickPoint(x=330, y=190, frame=10, label=1, object_id="moving_object_2"),
        ClickPoint(x=320, y=195, frame=15, label=1, object_id="moving_object_2"),
        
        # Object 3 - Stationary object (e.g., coral or rock)
        ClickPoint(x=250, y=280, frame=0, label=1, object_id="static_object_1"),
        ClickPoint(x=250, y=280, frame=10, label=1, object_id="static_object_1"),
        ClickPoint(x=250, y=280, frame=20, label=1, object_id="static_object_1"),
        
        # Object 4 - Another moving element
        ClickPoint(x=450, y=150, frame=0, label=1, object_id="moving_object_3"),
        ClickPoint(x=430, y=160, frame=7, label=1, object_id="moving_object_3"),
        ClickPoint(x=410, y=170, frame=14, label=1, object_id="moving_object_3"),
        
        # Object 5 - Background element
        ClickPoint(x=100, y=250, frame=0, label=1, object_id="background_element"),
        ClickPoint(x=105, y=252, frame=10, label=1, object_id="background_element"),
        ClickPoint(x=110, y=254, frame=20, label=1, object_id="background_element"),
    ]
    
    # Configuration for highlighted/colored output
    config = SegmentationConfig(
        mask_type="highlighted",     # Show colored overlays (each object gets different color)
        annotation_type="mask",       # Use masks (not bounding boxes)
        output_video=True,           # Output as video
        video_fps=30,                # Maintain original frame rate
        output_frame_interval=1      # Process every frame
    )
    
    # Show tracking info
    unique_objects = list(set(p.object_id for p in click_points))
    print(f"\n🎨 Tracking {len(unique_objects)} objects (each with unique color):")
    for i, obj in enumerate(unique_objects, 1):
        points = [p for p in click_points if p.object_id == obj]
        print(f"   Color {i}: {obj} ({len(points)} tracking points)")
    
    print("\n🚀 Sending to Replicate for processing...")
    print("⏱️  This typically takes 1-2 minutes...")
    
    try:
        # Run segmentation
        result = segmenter.segment_video_advanced(
            video_path=video_path,
            click_points=click_points,
            config=config,
            output_path=output_path
        )
        
        print(f"\n✅ Segmentation complete!")
        print(f"📹 Segmented video saved to: {result}")
        print("\n🎨 Each object appears in a different color in the output video")
        
        # Open video automatically on macOS
        if sys.platform == "darwin" and Path(output_path).exists():
            print("🖥️  Opening video...")
            os.system(f"open '{output_path}'")
        
        return result
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        if "unauthorized" in str(e).lower():
            print("\n⚠️  Authentication issue. Check your API key.")
        elif "rate limit" in str(e).lower():
            print("\n⚠️  Rate limit reached. Wait a moment and try again.")
        else:
            print(f"\nFull error: {e}")
        raise


if __name__ == "__main__":
    segment_sea_video()