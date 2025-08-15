#!/usr/bin/env python3
"""
SAM2 segmentation with dense tracking points for continuous masks
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))
from backend.utils.sam2_api.video_segmentation import SAM2VideoSegmenter, ClickPoint, SegmentationConfig

load_dotenv('./backend/.env')

def segment_with_continuous_tracking():
    """Create continuous segmentation with dense tracking points"""
    
    api_key = os.environ.get("REPLICATE_API_KEY") or os.environ.get("REPLICATE_API_TOKEN")
    if api_key:
        os.environ["REPLICATE_API_TOKEN"] = api_key
        print("✅ Using Replicate API key")
    else:
        print("❌ No API key found")
        return
    
    video_path = "uploads/assets/videos/sea_small.mov"
    output_path = "uploads/assets/videos/sea_small_continuous.mp4"
    
    print("\n🌊 SAM2 Continuous Segmentation")
    print(f"📹 Input: {video_path}")
    print(f"📹 Output: {output_path}")
    print("📊 Strategy: Dense tracking points every 10 frames for smooth propagation")
    
    segmenter = SAM2VideoSegmenter()
    
    # Create DENSE tracking points for continuous segmentation
    # Place points every 10 frames (6 times per second at 60fps)
    click_points = []
    
    # Total frames: 222 (0-221)
    # We'll place points every 10 frames for each object
    
    # 1. Palm fronds - consistent position throughout
    for frame in range(0, 222, 10):  # Every 10 frames
        click_points.append(
            ClickPoint(x=320, y=50, frame=frame, label=1, object_id="palm_fronds")
        )
    
    # 2. Ocean water - track the moving water
    for frame in range(0, 222, 15):  # Every 15 frames
        click_points.append(
            ClickPoint(x=400, y=180, frame=frame, label=1, object_id="ocean")
        )
    
    # 3. Rocks - stationary but important
    for frame in range(0, 222, 20):  # Every 20 frames
        click_points.append(
            ClickPoint(x=200, y=230, frame=frame, label=1, object_id="rocks")
        )
    
    # 4. Beach sand
    for frame in range(0, 222, 20):  # Every 20 frames
        click_points.append(
            ClickPoint(x=100, y=320, frame=frame, label=1, object_id="sand")
        )
    
    # 5. Sky/clouds
    for frame in range(0, 222, 25):  # Every 25 frames
        click_points.append(
            ClickPoint(x=500, y=100, frame=frame, label=1, object_id="sky")
        )
    
    config = SegmentationConfig(
        mask_type="highlighted",
        annotation_type="mask",
        output_video=True,
        video_fps=60,
        output_frame_interval=1  # Process EVERY frame
    )
    
    # Show tracking density
    unique_objects = list(set(p.object_id for p in click_points))
    print(f"\n🎯 Dense tracking setup:")
    for obj in unique_objects:
        points = [p for p in click_points if p.object_id == obj]
        print(f"   • {obj}: {len(points)} tracking points (one every {222//len(points)} frames)")
    
    print(f"\n📍 Total tracking points: {len(click_points)}")
    print("🚀 Processing with dense tracking for continuous masks...")
    
    try:
        result = segmenter.segment_video_advanced(
            video_path=video_path,
            click_points=click_points,
            config=config,
            output_path=output_path
        )
        
        print(f"\n✅ Success! Continuous segmentation saved to: {result}")
        print("🎬 Masks should now appear continuously throughout the video")
        
        if sys.platform == "darwin" and Path(output_path).exists():
            os.system(f"open '{output_path}'")
        
        return result
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise


if __name__ == "__main__":
    segment_with_continuous_tracking()