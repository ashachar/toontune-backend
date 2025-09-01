#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SAM2 Video Tracking for head/face detection.
Uses SAM2's video tracking to maintain consistent head detection across frames.
"""

import cv2
import numpy as np
import subprocess
import os
import sys
import json
import time
import replicate
import requests
from io import BytesIO
from PIL import Image
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '..', '.env'))

# Add paths for SAM2 imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'utils', 'sam2_api'))

try:
    from video_segmentation import SAM2VideoSegmenter, ClickPoint, SegmentationConfig
    SAM2_API_AVAILABLE = True
except ImportError:
    print("Warning: SAM2 API not available")
    SAM2_API_AVAILABLE = False


def detect_initial_face_position(frame: np.ndarray) -> Tuple[int, int]:
    """
    Get initial click position for SAM2 tracking.
    For this video, we know the face is in the upper-center region.
    """
    h, w = frame.shape[:2]
    # Click on upper-center where the face typically is
    center_x = w // 2
    center_y = h // 3
    print(f"  Using face position: ({center_x}, {center_y})")
    return center_x, center_y


def track_head_with_sam2_video(video_path: str, output_mask_video: str = None) -> str:
    """
    Track head/face throughout video using SAM2 video tracking.
    
    Args:
        video_path: Path to input video
        output_mask_video: Optional path to save mask video
        
    Returns:
        Path to mask video showing tracked head regions
    """
    if not SAM2_API_AVAILABLE:
        print("❌ SAM2 API not available")
        return None
    
    if not os.environ.get("REPLICATE_API_TOKEN"):
        print("❌ No REPLICATE_API_TOKEN found in environment")
        print("   Please set: export REPLICATE_API_TOKEN='your_token_here'")
        return None
    
    print("\n🎯 Using SAM2 VIDEO TRACKING for head detection")
    print("   This tracks the head consistently across all frames")
    
    # Read first frame to find initial face position
    cap = cv2.VideoCapture(video_path)
    ret, first_frame = cap.read()
    if not ret:
        print("❌ Could not read video")
        cap.release()
        return None
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    print(f"\nVideo properties: {width}x{height}, {fps:.1f} fps, {total_frames} frames")
    
    # Detect initial face position
    print("\nDetecting initial face position...")
    face_x, face_y = detect_initial_face_position(first_frame)
    
    # We'll also add some additional click points around the face for better coverage
    click_points = [
        (face_x, face_y, 0),  # Center of face at frame 0
        (face_x - 30, face_y - 30, 0),  # Top-left for hair
        (face_x + 30, face_y - 30, 0),  # Top-right for hair
        (face_x, face_y - 50, 0),  # Top for hair/forehead
    ]
    
    print(f"\nRunning SAM2 video tracking with {len(click_points)} click points...")
    for i, (x, y, frame) in enumerate(click_points):
        print(f"  Point {i+1}: ({x}, {y}) at frame {frame}")
    
    # Initialize SAM2 video segmenter
    segmenter = SAM2VideoSegmenter()
    
    # Configure for mask output
    config = SegmentationConfig(
        mask_type="greenscreen",  # Green screen mask for easy processing
        output_video=True,
        video_fps=int(fps),
        annotation_type="mask"
    )
    
    # Generate unique output name
    timestamp = int(time.time())
    temp_output = f"/tmp/sam2_head_mask_{timestamp}.mp4"
    
    try:
        # Run SAM2 video tracking
        print("\n🚀 Starting SAM2 video tracking (this may take a moment)...")
        result = segmenter.segment_video_advanced(
            video_path,
            [ClickPoint(x=x, y=y, frame=f, label=1, object_id="head") 
             for x, y, f in click_points],
            config,
            temp_output
        )
        
        print(f"\n✅ SAM2 tracking complete!")
        
        # Download result if it's a URL
        if isinstance(result, str) and result.startswith('http'):
            print(f"   Downloading mask video from: {result[:50]}...")
            response = requests.get(result)
            with open(temp_output, 'wb') as f:
                f.write(response.content)
            mask_video_path = temp_output
        else:
            mask_video_path = result
        
        # Save to output path if specified
        if output_mask_video:
            if os.path.exists(mask_video_path):
                os.rename(mask_video_path, output_mask_video)
                mask_video_path = output_mask_video
        
        return mask_video_path
        
    except Exception as e:
        print(f"❌ SAM2 tracking failed: {e}")
        return None


def create_head_tracking_debug_video(input_video: str, mask_video: str, output_path: str):
    """
    Create a debug video showing SAM2 head tracking results.
    
    Args:
        input_video: Path to original video
        mask_video: Path to SAM2 mask video (green screen format)
        output_path: Output debug video path
    """
    print("\n📹 Creating debug visualization video...")
    
    # Open videos
    cap_orig = cv2.VideoCapture(input_video)
    cap_mask = cv2.VideoCapture(mask_video)
    
    # Get video properties
    fps = cap_orig.get(cv2.CAP_PROP_FPS)
    width = int(cap_orig.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap_orig.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap_orig.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create output writer (side-by-side visualization)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width * 2, height))
    
    print(f"Processing {total_frames} frames...")
    
    # Track statistics
    head_coverage_stats = []
    
    for frame_idx in range(total_frames):
        ret_orig, frame_original = cap_orig.read()
        ret_mask, frame_mask = cap_mask.read()
        
        if not ret_orig or not ret_mask:
            break
        
        # Extract head mask from green screen video
        # SAM2 outputs green where the object is NOT present
        green_screen_color = np.array([0, 255, 0], dtype=np.uint8)
        tolerance = 50
        
        # Convert mask frame to RGB if needed
        if len(frame_mask.shape) == 2:
            frame_mask = cv2.cvtColor(frame_mask, cv2.COLOR_GRAY2BGR)
        
        # Detect green screen (background)
        diff = np.abs(frame_mask.astype(np.int16) - green_screen_color.astype(np.int16))
        is_green = np.all(diff <= tolerance, axis=2)
        
        # Head mask is NOT green
        head_mask = (~is_green).astype(np.uint8) * 255
        
        # Create visualizations
        # 1. Original with overlay
        overlay = frame_original.copy()
        red_mask = np.zeros_like(frame_original)
        red_mask[:, :, 2] = head_mask  # Red channel
        overlay = cv2.addWeighted(overlay, 0.7, red_mask, 0.3, 0)
        
        # 2. Binary mask
        mask_3ch = cv2.cvtColor(head_mask, cv2.COLOR_GRAY2BGR)
        
        # Add text labels
        cv2.putText(overlay, "Original + Head Tracking", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(mask_3ch, "SAM2 Tracked Head Mask", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Add statistics
        head_pixels = np.count_nonzero(head_mask)
        total_pixels = width * height
        coverage = (head_pixels / total_pixels) * 100
        head_coverage_stats.append(coverage)
        
        stats_text = f"Frame {frame_idx}/{total_frames} | Head: {head_pixels:,} px ({coverage:.1f}%)"
        cv2.putText(overlay, stats_text, (10, height - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(mask_3ch, stats_text, (10, height - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Add tracking quality indicator
        quality_text = "Tracking: STABLE" if coverage > 1 else "Tracking: LOST"
        quality_color = (0, 255, 0) if coverage > 1 else (0, 0, 255)
        cv2.putText(overlay, quality_text, (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, quality_color, 2)
        
        # Combine side by side
        combined = np.hstack([overlay, mask_3ch])
        out.write(combined)
        
        if frame_idx % 30 == 0:
            print(f"  Processed {frame_idx}/{total_frames} frames (coverage: {coverage:.1f}%)")
    
    # Clean up
    cap_orig.release()
    cap_mask.release()
    out.release()
    cv2.destroyAllWindows()
    
    # Print statistics
    if head_coverage_stats:
        avg_coverage = np.mean(head_coverage_stats)
        min_coverage = np.min(head_coverage_stats)
        max_coverage = np.max(head_coverage_stats)
        
        print(f"\n📊 Head tracking statistics:")
        print(f"  Average coverage: {avg_coverage:.1f}%")
        print(f"  Min coverage: {min_coverage:.1f}%")
        print(f"  Max coverage: {max_coverage:.1f}%")
        print(f"  Tracking stability: {sum(c > 1 for c in head_coverage_stats) / len(head_coverage_stats) * 100:.1f}%")
    
    print(f"\nDebug video saved: {output_path}")
    
    # Convert to H.264
    output_h264 = output_path.replace('.mp4', '_h264.mp4')
    cmd = [
        "ffmpeg", "-y",
        "-i", output_path,
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        output_h264
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    print(f"H.264 version: {output_h264}")
    
    # Remove temp file
    os.remove(output_path)
    return output_h264


def main():
    input_video = "ai_math1_6sec.mp4"
    
    # Check for Replicate API token
    if not os.environ.get("REPLICATE_API_TOKEN"):
        print("\n❌ ERROR: REPLICATE_API_TOKEN not found in environment")
        print("   SAM2 video tracking requires Replicate API access")
        print("\n   Please set your token:")
        print("   export REPLICATE_API_TOKEN='your_token_here'")
        print("\n   Get your token at: https://replicate.com/account/api-tokens")
        return
    
    print("\n" + "="*60)
    print("SAM2 VIDEO TRACKING HEAD DETECTION")
    print("="*60)
    print("\nThis uses SAM2's video tracking model to:")
    print("  • Track head/face consistently across all frames")
    print("  • No flickering or frame-to-frame inconsistency")
    print("  • Better quality than per-frame detection")
    
    # Step 1: Run SAM2 video tracking
    mask_video = track_head_with_sam2_video(
        input_video,
        output_mask_video="sam2_tracked_head_mask.mp4"
    )
    
    if not mask_video:
        print("\n❌ Failed to generate head tracking mask")
        return
    
    # Step 2: Create debug visualization
    debug_video = create_head_tracking_debug_video(
        input_video,
        mask_video,
        "sam2_head_tracking_debug.mp4"
    )
    
    print(f"\n✅ SAM2 head tracking complete!")
    print(f"\n📁 Output files:")
    print(f"  • Mask video: {mask_video}")
    print(f"  • Debug video: {debug_video}")
    
    print("\n🎯 Key advantages over image segmentation:")
    print("  • Temporal consistency (no flickering)")
    print("  • Better handling of occlusions")
    print("  • Smoother mask boundaries")
    print("  • More efficient (one-time tracking)")


if __name__ == "__main__":
    main()