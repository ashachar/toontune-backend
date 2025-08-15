#!/usr/bin/env python3
"""
SAM2 segmentation with points on every frame for truly continuous masks
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))
from backend.utils.sam2_api.video_segmentation import SAM2VideoSegmenter, ClickPoint, SegmentationConfig

load_dotenv('./backend/.env')

def segment_every_frame():
    """Place tracking points on EVERY frame for key objects"""
    
    api_key = os.environ.get("REPLICATE_API_KEY") or os.environ.get("REPLICATE_API_TOKEN")
    if api_key:
        os.environ["REPLICATE_API_TOKEN"] = api_key
        print("✅ Using Replicate API key")
    else:
        return
    
    video_path = "uploads/assets/videos/sea_small.mov"
    output_path = "uploads/assets/videos/sea_small_every_frame.mp4"
    
    print("\n🌊 SAM2 Every-Frame Segmentation")
    print("📊 Strategy: Track 2 main objects on EVERY frame (222 frames)")
    
    segmenter = SAM2VideoSegmenter()
    click_points = []
    
    # Track just 2 main objects but on EVERY frame
    # This ensures continuous masks but limits API load
    
    for frame in range(0, 222, 1):  # Every single frame
        # Palm fronds - most visible object
        click_points.append(
            ClickPoint(x=320, y=50, frame=frame, label=1, object_id="palm_fronds")
        )
        
        # Ocean water - second most important
        click_points.append(
            ClickPoint(x=400, y=180, frame=frame, label=1, object_id="ocean")
        )
    
    # Add sparse points for other objects
    for frame in range(0, 222, 30):  # Every 0.5 seconds
        click_points.append(
            ClickPoint(x=200, y=230, frame=frame, label=1, object_id="rocks")
        )
    
    config = SegmentationConfig(
        mask_type="highlighted",
        annotation_type="mask",
        output_video=True,
        video_fps=60,
        output_frame_interval=1
    )
    
    print(f"\n📍 Total tracking points: {len(click_points)}")
    print("   • palm_fronds: 222 points (every frame)")
    print("   • ocean: 222 points (every frame)")
    print("   • rocks: 8 points (every 30 frames)")
    print("\n⚠️  Note: This may take longer due to many tracking points")
    
    try:
        result = segmenter.segment_video_advanced(
            video_path=video_path,
            click_points=click_points,
            config=config,
            output_path=output_path
        )
        
        print(f"\n✅ Success! Every-frame segmentation saved to: {result}")
        
        if sys.platform == "darwin" and Path(output_path).exists():
            os.system(f"open '{output_path}'")
        
        return result
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise


if __name__ == "__main__":
    segment_every_frame()