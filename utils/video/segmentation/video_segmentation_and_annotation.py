#!/usr/bin/env python3
"""
Video Segmentation and Annotation Pipeline

This pipeline automatically segments video frames and generates descriptive labels for each segment.
It uses SAM2 for segmentation and Gemini for generating natural language descriptions.

Key features:
- Automatic segmentation using SAM2 (no manual clicks required)
- Two-asset protocol to avoid color confusion in descriptions
- Tracks up to 10 largest segments through the video
- Generates concise 2-4 word labels for each segment
- Outputs labeled video with segment overlays and text annotations

Usage:
    python video_segmentation_and_annotation.py <input_video_path> [output_dir]
"""

import time
import cv2
import numpy as np
from pathlib import Path
import replicate
import json
import requests
from PIL import Image
from io import BytesIO
import os
from dotenv import load_dotenv
import subprocess
import google.generativeai as genai
import sys

# Load environment variables
load_dotenv()

def setup_gemini():
    """Setup Gemini API"""
    gemini_key = os.environ.get("GEMINI_API_KEY")
    if not gemini_key:
        print("❌ No GEMINI_API_KEY found in .env")
        sys.exit(1)
    
    genai.configure(api_key=gemini_key)
    model = genai.GenerativeModel('gemini-2.0-flash-exp')
    print("✅ Gemini configured")
    return model

def check_replicate():
    """Check Replicate API token"""
    if not os.environ.get("REPLICATE_API_TOKEN"):
        print("❌ No REPLICATE_API_TOKEN found")
        sys.exit(1)
    print("✅ Replicate token loaded")

def segment_first_frame(frame_path, max_retries=3):
    """
    Run SAM2 segmentation on the first frame
    Returns list of segment dictionaries with masks and metadata
    """
    print(f"\n[{time.strftime('%H:%M:%S')}] Running SAM2 segmentation...")
    
    prediction = None
    for attempt in range(max_retries):
        try:
            print(f"[{time.strftime('%H:%M:%S')}] Attempt {attempt + 1}/{max_retries}...")
            with open(frame_path, 'rb') as f:
                prediction = replicate.run(
                    'meta/sam-2:fe97b453a6455861e3bac769b441ca1f1086110da7466dbb65cf1eecfd60dc83',
                    input={
                        'image': f,
                        'points_per_side': 32,
                        'pred_iou_thresh': 0.7,
                        'stability_score_thresh': 0.85,
                        'use_m2m': True,
                        'multimask_output': False
                    }
                )
            break
        except Exception as e:
            print(f"[{time.strftime('%H:%M:%S')}] Attempt {attempt + 1} failed: {e}")
            if attempt == max_retries - 1:
                print("All attempts failed.")
                return None
            time.sleep(5)
    
    if not prediction:
        return None
    
    # Get individual masks
    individual_masks_urls = prediction.get('individual_masks', [])
    print(f"[{time.strftime('%H:%M:%S')}] Found {len(individual_masks_urls)} masks")
    
    # Process masks
    segments = []
    frame = cv2.imread(str(frame_path))
    
    for i, mask_url in enumerate(individual_masks_urls):
        response = requests.get(str(mask_url))
        mask_img = Image.open(BytesIO(response.content))
        mask_array = np.array(mask_img)
        
        if len(mask_array.shape) == 3:
            mask_array = cv2.cvtColor(mask_array, cv2.COLOR_RGB2GRAY)
        
        if mask_array.shape[:2] != frame.shape[:2]:
            mask_array = cv2.resize(mask_array, (frame.shape[1], frame.shape[0]))
        
        binary_mask = (mask_array > 128).astype(np.uint8)
        area = np.sum(binary_mask)
        
        if area < 50:  # Skip tiny segments
            continue
        
        y_coords, x_coords = np.where(binary_mask)
        cx = int(np.mean(x_coords))
        cy = int(np.mean(y_coords))
        
        segments.append({
            'id': len(segments),
            'mask': binary_mask,
            'area': area,
            'centroid': (cx, cy)
        })
    
    # Sort by area and take top 10
    segments = sorted(segments, key=lambda x: x['area'], reverse=True)[:10]
    print(f"[{time.strftime('%H:%M:%S')}] Using top {len(segments)} segments")
    
    # Re-number segments
    for i, seg in enumerate(segments):
        seg['id'] = i + 1
    
    return segments

def create_two_asset_image(frame, segments, output_dir):
    """
    Create concatenated image for Gemini: original + grayscale segment mask
    """
    print(f"\n[{time.strftime('%H:%M:%S')}] Creating two-asset image...")
    
    # Create grayscale mask where pixel value = segment ID
    segment_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    
    for seg in segments:
        segment_mask[seg['mask'] > 0] = seg['id']
    
    # Scale up for visibility
    segment_mask_vis = segment_mask * 25
    
    # Add text labels
    mask_with_labels = cv2.cvtColor(segment_mask_vis, cv2.COLOR_GRAY2BGR)
    for seg in segments:
        cx, cy = seg['centroid']
        cv2.circle(mask_with_labels, (cx, cy), 20, (255, 255, 255), -1)
        cv2.putText(mask_with_labels, str(seg['id']),
                   (cx - 10, cy + 7),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    
    # Create side-by-side image
    height, width = frame.shape[:2]
    concatenated = np.zeros((height, width * 2, 3), dtype=np.uint8)
    concatenated[:, :width] = frame
    concatenated[:, width:] = mask_with_labels
    
    # Add dividing line
    cv2.line(concatenated, (width, 0), (width, height), (255, 255, 255), 2)
    
    # Save
    concat_path = output_dir / "concatenated_input.png"
    cv2.imwrite(str(concat_path), concatenated)
    
    return concat_path

def get_segment_descriptions(concat_path, segments, model):
    """
    Use Gemini to get descriptions for each segment
    """
    print(f"\n[{time.strftime('%H:%M:%S')}] Getting segment descriptions from Gemini...")
    
    gemini_image = Image.open(concat_path)
    
    prompt = """This image contains two parts side by side:
    
LEFT: The original photograph
RIGHT: A symbolic segment mask where each distinct gray level represents a segment ID (numbered 1-10)

The right image is purely symbolic - ignore its visual appearance and colors. The numbers shown indicate segment IDs.
Each gray level in the right image corresponds to a different segment in the left image.

Analyze what is visible in each numbered segment of the LEFT image.
DO NOT mention colors, as they are artificially added for visualization.
Focus only on what objects or features are present in each segment.

Provide your response as valid JSON with segment IDs as keys.
Each description should be CONCISE: just 2-4 words describing what is in that segment.

Example format:
{
    "1": "ocean water",
    "2": "clear sky",
    "3": "palm trees",
    ...and so on for all visible segment IDs
}

IMPORTANT: Keep descriptions SHORT (2-4 words). Do not start with "This segment" or similar phrases.
Describe only the actual content without mentioning any colors."""
    
    try:
        response = model.generate_content([prompt, gemini_image])
        gemini_output = response.text
        
        print(f"[{time.strftime('%H:%M:%S')}] Gemini response received")
        
        # Parse JSON
        if "```json" in gemini_output:
            gemini_output = gemini_output.split("```json")[1].split("```")[0]
        elif "```" in gemini_output:
            gemini_output = gemini_output.split("```")[1].split("```")[0]
        
        descriptions = json.loads(gemini_output.strip())
        
        # Assign to segments
        for seg in segments:
            seg_id = str(seg['id'])
            if seg_id in descriptions:
                seg['description'] = descriptions[seg_id]
            else:
                seg['description'] = f"Segment {seg_id}"
                
    except Exception as e:
        print(f"Gemini error: {e}")
        for seg in segments:
            seg['description'] = f"Segment {seg['id']}"
    
    return segments

def create_labeled_frame(frame, segments, output_dir):
    """
    Create a labeled visualization of the first frame
    """
    print(f"\n[{time.strftime('%H:%M:%S')}] Creating labeled frame...")
    
    final_vis = frame.copy()
    
    # Apply colored overlays
    colors = [(255,100,100), (100,255,100), (100,100,255), (255,255,100), 
              (255,100,255), (100,255,255), (255,200,100), (200,100,255),
              (100,200,255), (255,255,200)]
    
    for i, seg in enumerate(segments):
        color_overlay = np.zeros_like(frame)
        color_overlay[seg['mask'] > 0] = colors[i % len(colors)]
        final_vis = cv2.addWeighted(final_vis, 1, color_overlay, 0.3, 0)
    
    # Add labels
    for seg in segments:
        cx, cy = seg['centroid']
        desc = seg.get('description', f"Segment {seg['id']}")
        
        if len(desc) > 30:
            desc = desc[:27] + "..."
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4
        thickness = 1
        text_size, _ = cv2.getTextSize(desc, font, font_scale, thickness)
        
        padding = 5
        cv2.rectangle(final_vis,
                    (cx - text_size[0]//2 - padding, cy - text_size[1] - padding),
                    (cx + text_size[0]//2 + padding, cy + padding),
                    (0, 0, 0), -1)
        
        cv2.putText(final_vis, desc,
                   (cx - text_size[0]//2, cy),
                   font, font_scale,
                   (255, 255, 255), thickness, cv2.LINE_AA)
    
    final_path = output_dir / "labeled_frame.png"
    cv2.imwrite(str(final_path), final_vis)
    
    return final_path

def track_segments_in_video(video_path, segments, output_dir):
    """
    Use SAM2 video model to track segments throughout the video
    """
    print(f"\n[{time.strftime('%H:%M:%S')}] Tracking segments in video...")
    
    coords = ",".join([f"[{s['centroid'][0]},{s['centroid'][1]}]" for s in segments])
    ids = ",".join([f"seg_{s['id']}" for s in segments])
    
    with open(video_path, 'rb') as video_file:
        output_video = replicate.run(
            "meta/sam-2-video:33432afdfc06a10da6b4018932893d39b0159f838b6d11dd1236dff85cc5ec1d",
            input={
                "input_video": video_file,
                "click_frames": "0",
                "click_coordinates": coords,
                "click_object_ids": ids,
                "mask_type": "highlighted",
                "output_video": True
            }
        )
    
    if hasattr(output_video, '__iter__'):
        outputs = list(output_video)
        mask_video_url = outputs[-1] if outputs else None
    else:
        mask_video_url = str(output_video)
    
    print(f"[{time.strftime('%H:%M:%S')}] Downloading tracked video...")
    response = requests.get(mask_video_url)
    mask_video_path = output_dir / "tracked_video.mp4"
    with open(mask_video_path, 'wb') as f:
        f.write(response.content)
    
    return mask_video_path

def create_final_labeled_video(video_path, mask_video_path, segments, output_dir):
    """
    Create final video with labels
    """
    print(f"\n[{time.strftime('%H:%M:%S')}] Creating final labeled video...")
    print(f"[{time.strftime('%H:%M:%S')}] Adding labels for {len(segments)} segments")
    
    cap_orig = cv2.VideoCapture(str(video_path))
    cap_mask = cv2.VideoCapture(str(mask_video_path))
    
    fps = int(cap_orig.get(cv2.CAP_PROP_FPS))
    width = int(cap_orig.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap_orig.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    output_path = output_dir / "labeled_video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    frame_count = 0
    while True:
        ret1, frame1 = cap_orig.read()
        ret2, frame2 = cap_mask.read()
        if not ret1 or not ret2:
            break
        
        # Blend
        result = cv2.addWeighted(frame1, 0.6, frame2, 0.4, 0)
        
        # Add labels
        for seg in segments:
            cx, cy = seg['centroid']
            desc = seg.get('description', '')
            
            if len(desc) > 25:
                desc = desc[:22] + "..."
            
            if desc:
                text_size = cv2.getTextSize(desc, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
                cv2.rectangle(result,
                            (cx - text_size[0]//2 - 3, cy - text_size[1] - 3),
                            (cx + text_size[0]//2 + 3, cy + 3),
                            (0, 0, 0), -1)
                cv2.putText(result, desc,
                           (cx - text_size[0]//2, cy),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
        
        out.write(result)
        frame_count += 1
    
    cap_orig.release()
    cap_mask.release()
    out.release()
    
    # Convert to H.264
    h264_path = output_dir / "final_video_h264.mp4"
    subprocess.run(['ffmpeg', '-i', str(output_path), '-c:v', 'libx264', '-y', str(h264_path)],
                   capture_output=True)
    
    print(f"[{time.strftime('%H:%M:%S')}] Processed {frame_count} frames")
    
    return h264_path

def print_final_report(segments, output_dir, h264_path):
    """
    Print final pipeline report
    """
    print(f"\n{'='*60}")
    print(f"✅ PIPELINE COMPLETE - FINAL REPORT")
    print(f"{'='*60}")
    print(f"\nSegments found: {len(segments)}")
    print(f"\nSegment descriptions:")
    for seg in segments:
        print(f"  {seg['id']}. {seg.get('description', 'No description')} (area: {seg['area']})")
    print(f"\nOutput files:")
    print(f"  - Concatenated input: {output_dir}/concatenated_input.png")
    print(f"  - Labeled frame: {output_dir}/labeled_frame.png")
    print(f"  - Final video: {h264_path}")
    print(f"{'='*60}")

def main():
    print("[START] Video Segmentation and Annotation Pipeline")
    
    # Setup
    model = setup_gemini()
    check_replicate()
    
    # Parse arguments
    if len(sys.argv) < 2:
        print("Usage: python video_segmentation_and_annotation.py <input_video> [output_dir]")
        sys.exit(1)
    
    video_path = Path(sys.argv[1])
    if not video_path.exists():
        print(f"Error: Video file not found: {video_path}")
        sys.exit(1)
    
    output_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("output/video_segmentation")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract first frame
    print(f"\n[{time.strftime('%H:%M:%S')}] Extracting first frame...")
    cap = cv2.VideoCapture(str(video_path))
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("Failed to read video")
        sys.exit(1)
    
    frame_path = output_dir / "frame0.png"
    cv2.imwrite(str(frame_path), frame)
    
    # Step 1: Segment first frame
    segments = segment_first_frame(frame_path)
    if not segments:
        print("Segmentation failed")
        sys.exit(1)
    
    # Step 2: Create two-asset image
    concat_path = create_two_asset_image(frame, segments, output_dir)
    
    # Step 3: Get descriptions from Gemini
    segments = get_segment_descriptions(concat_path, segments, model)
    
    # Step 4: Create labeled frame
    labeled_frame_path = create_labeled_frame(frame, segments, output_dir)
    
    # Step 5: Track segments in video
    mask_video_path = track_segments_in_video(video_path, segments, output_dir)
    
    # Step 6: Create final labeled video
    h264_path = create_final_labeled_video(video_path, mask_video_path, segments, output_dir)
    
    # Print report
    print_final_report(segments, output_dir, h264_path)

if __name__ == "__main__":
    main()