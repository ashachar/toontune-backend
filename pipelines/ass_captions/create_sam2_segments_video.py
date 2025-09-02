#!/usr/bin/env python3
"""
Create a video showing SAM2 segments for visualization.
Uses automatic mask generation on key frames.
"""

import cv2
import numpy as np
import json
import os
from pathlib import Path
import replicate
from dotenv import load_dotenv

load_dotenv()

def create_sam2_segments_video(input_video: str, output_path: str = None):
    """
    Create a video showing SAM2 automatic segmentation.
    """
    video_path = Path(input_video)
    if not video_path.exists():
        print(f"Error: Video not found: {video_path}")
        return None
    
    if output_path is None:
        output_path = f"../../outputs/{video_path.stem}_sam2_segments.mp4"
    
    print(f"\n🎯 Creating SAM2 segmentation video for: {input_video}")
    
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print("Error: Cannot open video")
        return None
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"   Video: {width}x{height}, {total_frames} frames")
    
    # Sample frames for segmentation (every 30 frames)
    sample_interval = 30
    segment_masks = {}
    
    # Process sampled frames with SAM2
    frame_idx = 0
    sampled_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx % sample_interval == 0 and sampled_count < 10:  # Limit to 10 samples
            print(f"   Segmenting frame {frame_idx}...")
            
            # Save frame temporarily
            temp_frame = f"/tmp/frame_{frame_idx}.jpg"
            cv2.imwrite(temp_frame, frame)
            
            try:
                # Use SAM2 automatic mask generator
                output = replicate.run(
                    "zsxkib/segment-anything-2:3a96c8c4fd0e8c6a1fb86c5e2e85f413cb2fabe33bfc901d36a27b4b7cbee670",
                    input={
                        "image": open(temp_frame, "rb"),
                        "mode": "automatic",
                        "points_per_side": 16,  # Reduced for faster processing
                        "pred_iou_thresh": 0.86,
                        "stability_score_thresh": 0.92,
                        "min_mask_region_area": 100
                    }
                )
                
                # Process masks
                if output and 'masks' in output:
                    masks = output['masks']
                    segment_masks[frame_idx] = process_masks(masks, width, height)
                    print(f"      Found {len(masks)} segments")
                
            except Exception as e:
                print(f"      Segmentation failed: {e}")
            
            # Clean up
            if os.path.exists(temp_frame):
                os.remove(temp_frame)
            
            sampled_count += 1
        
        frame_idx += 1
    
    # Reset video to create output
    cap.release()
    cap = cv2.VideoCapture(str(video_path))
    
    # Find nearest segmentation for interpolation
    sorted_frames = sorted(segment_masks.keys())
    
    print(f"\n   Creating segmented video...")
    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Find nearest segmentation
        nearest_seg = find_nearest_segmentation(frame_idx, sorted_frames)
        
        if nearest_seg is not None and nearest_seg in segment_masks:
            # Apply segmentation overlay
            segments = segment_masks[nearest_seg]
            overlay = create_segment_overlay(frame, segments)
            
            # Blend with original
            alpha = 0.6
            result = cv2.addWeighted(frame, 1-alpha, overlay, alpha, 0)
            
            # Add text showing segment info
            cv2.putText(result, f"Frame {frame_idx} | Segments: {len(segments)}", 
                       (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        else:
            result = frame
        
        out.write(result)
        
        if frame_idx % 100 == 0:
            print(f"      Processed {frame_idx}/{total_frames} frames")
        
        frame_idx += 1
    
    cap.release()
    out.release()
    
    print(f"\n✅ SAM2 segments video created: {output_path}")
    
    # Convert to H.264
    output_h264 = output_path.replace('.mp4', '_h264.mp4')
    cmd = [
        'ffmpeg', '-i', output_path,
        '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
        '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
        '-y', output_h264
    ]
    
    import subprocess
    subprocess.run(cmd, capture_output=True)
    os.remove(output_path)
    
    print(f"✅ H.264 version: {output_h264}")
    return output_h264


def process_masks(masks, width, height):
    """Process SAM2 masks into colored segments."""
    segments = []
    colors = generate_distinct_colors(len(masks))
    
    for i, mask_data in enumerate(masks):
        # Convert mask to binary
        mask = np.array(mask_data['segmentation'])
        
        # Get bounding box
        coords = np.where(mask)
        if len(coords[0]) > 0:
            y_min, y_max = coords[0].min(), coords[0].max()
            x_min, x_max = coords[1].min(), coords[1].max()
            
            segments.append({
                'mask': mask,
                'color': colors[i],
                'bbox': (x_min, y_min, x_max, y_max),
                'area': np.sum(mask)
            })
    
    return segments


def generate_distinct_colors(n):
    """Generate visually distinct colors for segments."""
    colors = []
    for i in range(n):
        hue = i * 360 / n
        # Convert HSV to BGR
        color = cv2.cvtColor(np.uint8([[[hue/2, 255, 255]]]), cv2.COLOR_HSV2BGR)[0][0]
        colors.append(color.tolist())
    return colors


def find_nearest_segmentation(frame_idx, sorted_frames):
    """Find the nearest frame with segmentation."""
    if not sorted_frames:
        return None
    
    # Binary search for nearest
    import bisect
    pos = bisect.bisect_left(sorted_frames, frame_idx)
    
    if pos == 0:
        return sorted_frames[0]
    if pos == len(sorted_frames):
        return sorted_frames[-1]
    
    # Choose closer one
    before = sorted_frames[pos - 1]
    after = sorted_frames[pos]
    
    if abs(frame_idx - before) < abs(frame_idx - after):
        return before
    return after


def create_segment_overlay(frame, segments):
    """Create colored overlay from segments."""
    overlay = np.zeros_like(frame)
    
    for seg in segments:
        mask = seg['mask']
        color = seg['color']
        
        # Apply color to masked region
        overlay[mask] = color
    
    return overlay


if __name__ == "__main__":
    # Test with first 30 seconds of video
    import subprocess
    
    # Create a short test video
    print("Creating 30-second test video...")
    test_input = "../../uploads/assets/videos/ai_math1.mp4"
    test_30s = "/tmp/ai_math1_30s.mp4"
    
    subprocess.run([
        'ffmpeg', '-i', test_input,
        '-t', '30', '-c', 'copy',
        '-y', test_30s
    ], capture_output=True)
    
    # Create segments video
    output = create_sam2_segments_video(test_30s)
    
    if output:
        print(f"\n✨ SAM2 segments visualization ready: {output}")
        print("   This shows automatic segmentation regions detected by SAM2")
