#!/usr/bin/env python3
"""
Quick script to run SAM2 segmentation on sea_small.mov
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.utils.sam2_api.video_segmentation import SAM2VideoSegmenter, ClickPoint, SegmentationConfig


def main():
    # Check for API token
    if not os.environ.get("REPLICATE_API_TOKEN"):
        print("=" * 60)
        print("⚠️  REPLICATE_API_TOKEN not set!")
        print("=" * 60)
        print("\nTo use SAM2 video segmentation, you need a Replicate API token.")
        print("\n1. Get your token at: https://replicate.com/account/api-tokens")
        print("2. Set it as an environment variable:")
        print("   export REPLICATE_API_TOKEN='your_token_here'")
        print("\nOnce you have the token, run this script again.")
        print("=" * 60)
        
        # Show what the script would do
        print("\n📋 What this script will do:")
        print("1. Load the video: uploads/assets/videos/sea_small.mov")
        print("2. Define click points on objects to segment")
        print("3. Send to SAM2 model on Replicate for processing")
        print("4. Download and save the segmented video")
        print("\n🎥 The output will be a video with highlighted segments")
        return
    
    # Video paths
    video_path = "uploads/assets/videos/sea_small.mov"
    output_path = "uploads/assets/videos/sea_small_segmented.mp4"
    
    print("🚀 Starting SAM2 Video Segmentation")
    print(f"Input: {video_path}")
    print(f"Output: {output_path}")
    
    # Initialize segmenter
    segmenter = SAM2VideoSegmenter()
    
    # Define click points for segmentation
    # These are example coordinates - adjust based on your video content
    click_points = [
        # Frame 0 - initial objects
        ClickPoint(x=150, y=100, frame=0, label=1, object_id="object_1"),
        ClickPoint(x=250, y=150, frame=0, label=1, object_id="object_2"),
        
        # Frame 10 - track objects
        ClickPoint(x=180, y=120, frame=10, label=1, object_id="object_1"),
        ClickPoint(x=280, y=170, frame=10, label=1, object_id="object_2"),
        
        # Frame 20 - continue tracking
        ClickPoint(x=200, y=140, frame=20, label=1, object_id="object_1"),
        ClickPoint(x=300, y=190, frame=20, label=1, object_id="object_2"),
    ]
    
    # Configuration
    config = SegmentationConfig(
        mask_type="highlighted",     # Show colored overlay
        annotation_type="mask",       # Show mask (not bounding box)
        output_video=True,           # Output as video file
        video_fps=30,                # Frame rate
        output_frame_interval=1      # Process every frame
    )
    
    print(f"\n📍 Segmenting {len(set(p.object_id for p in click_points))} objects")
    
    try:
        # Run segmentation
        result = segmenter.segment_video_advanced(
            video_path=video_path,
            click_points=click_points,
            config=config,
            output_path=output_path
        )
        
        print(f"\n✅ Success! Segmented video saved to: {result}")
        
        # Open the video on macOS
        if sys.platform == "darwin" and Path(output_path).exists():
            os.system(f"open '{output_path}'")
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        

if __name__ == "__main__":
    main()