#!/usr/bin/env python3
"""
Fixed SAM2 segmentation for sea_small.mov with proper frame distribution
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))
from backend.utils.sam2_api.video_segmentation import SAM2VideoSegmenter, ClickPoint, SegmentationConfig

# Load environment variables
load_dotenv('./backend/.env')

def segment_sea_properly():
    """Segment sea video with click points distributed across entire duration"""
    
    # Set API token
    api_key = os.environ.get("REPLICATE_API_KEY") or os.environ.get("REPLICATE_API_TOKEN")
    if api_key:
        os.environ["REPLICATE_API_TOKEN"] = api_key
        print("✅ Using Replicate API key from .env file")
    else:
        print("❌ No Replicate API key found")
        return
    
    video_path = "uploads/assets/videos/sea_small.mov"
    output_path = "uploads/assets/videos/sea_small_segmented_fixed.mp4"
    
    print("\n🌊 SAM2 Segmentation - Fixed Frame Distribution")
    print(f"📹 Input: {video_path}")
    print(f"📹 Output: {output_path}")
    print("\n📊 Video info: 60fps, 222 frames, 3.7 seconds")
    
    segmenter = SAM2VideoSegmenter()
    
    # Define click points distributed across the ENTIRE video
    # Video has 222 frames at 60fps (3.7 seconds)
    click_points = [
        # Palm fronds (top of frame) - tracked throughout video
        ClickPoint(x=320, y=50, frame=0, label=1, object_id="palm_fronds"),
        ClickPoint(x=320, y=50, frame=30, label=1, object_id="palm_fronds"),   # 0.5s
        ClickPoint(x=320, y=50, frame=60, label=1, object_id="palm_fronds"),   # 1.0s
        ClickPoint(x=320, y=50, frame=90, label=1, object_id="palm_fronds"),   # 1.5s
        ClickPoint(x=320, y=50, frame=120, label=1, object_id="palm_fronds"),  # 2.0s
        ClickPoint(x=320, y=50, frame=150, label=1, object_id="palm_fronds"),  # 2.5s
        ClickPoint(x=320, y=50, frame=180, label=1, object_id="palm_fronds"),  # 3.0s
        ClickPoint(x=320, y=50, frame=210, label=1, object_id="palm_fronds"),  # 3.5s
        
        # Rocks in water - tracked throughout
        ClickPoint(x=200, y=230, frame=0, label=1, object_id="rocks"),
        ClickPoint(x=200, y=230, frame=40, label=1, object_id="rocks"),
        ClickPoint(x=200, y=230, frame=80, label=1, object_id="rocks"),
        ClickPoint(x=200, y=230, frame=120, label=1, object_id="rocks"),
        ClickPoint(x=200, y=230, frame=160, label=1, object_id="rocks"),
        ClickPoint(x=200, y=230, frame=200, label=1, object_id="rocks"),
        
        # Ocean water - tracked throughout
        ClickPoint(x=400, y=180, frame=0, label=1, object_id="ocean"),
        ClickPoint(x=400, y=180, frame=50, label=1, object_id="ocean"),
        ClickPoint(x=400, y=180, frame=100, label=1, object_id="ocean"),
        ClickPoint(x=400, y=180, frame=150, label=1, object_id="ocean"),
        ClickPoint(x=400, y=180, frame=200, label=1, object_id="ocean"),
        
        # Beach sand - tracked throughout
        ClickPoint(x=100, y=320, frame=0, label=1, object_id="sand"),
        ClickPoint(x=100, y=320, frame=60, label=1, object_id="sand"),
        ClickPoint(x=100, y=320, frame=120, label=1, object_id="sand"),
        ClickPoint(x=100, y=320, frame=180, label=1, object_id="sand"),
        
        # Sky/clouds - tracked throughout
        ClickPoint(x=500, y=100, frame=0, label=1, object_id="sky"),
        ClickPoint(x=500, y=100, frame=70, label=1, object_id="sky"),
        ClickPoint(x=500, y=100, frame=140, label=1, object_id="sky"),
        ClickPoint(x=500, y=100, frame=210, label=1, object_id="sky"),
        
        # Log on beach - stationary object
        ClickPoint(x=80, y=280, frame=0, label=1, object_id="log"),
        ClickPoint(x=80, y=280, frame=110, label=1, object_id="log"),
        ClickPoint(x=80, y=280, frame=220, label=1, object_id="log"),
    ]
    
    config = SegmentationConfig(
        mask_type="highlighted",     # Colored overlays
        annotation_type="mask",       # Mask only
        output_video=True,
        video_fps=60,                # Match original fps
        output_frame_interval=1      # Process every frame
    )
    
    # Show tracking info
    unique_objects = list(set(p.object_id for p in click_points))
    print(f"\n🎨 Tracking {len(unique_objects)} objects across entire video:")
    for i, obj in enumerate(unique_objects, 1):
        points = [p for p in click_points if p.object_id == obj]
        frames = [p.frame for p in points]
        print(f"   Color {i}: {obj} - {len(points)} points (frames {min(frames)}-{max(frames)})")
    
    print("\n🚀 Processing with proper frame distribution...")
    
    try:
        result = segmenter.segment_video_advanced(
            video_path=video_path,
            click_points=click_points,
            config=config,
            output_path=output_path
        )
        
        print(f"\n✅ Success! Segmented video saved to: {result}")
        print("🎬 The segmentation should now persist throughout the entire video!")
        
        if sys.platform == "darwin" and Path(output_path).exists():
            print("🖥️  Opening video...")
            os.system(f"open '{output_path}'")
        
        return result
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise


if __name__ == "__main__":
    segment_sea_properly()