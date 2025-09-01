#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Debug script to visualize SAM2 head/face detection in video.
Shows detected head regions as white pixels on black background.
"""

import cv2
import numpy as np
import subprocess
import os
import sys
import json
from pathlib import Path

# Add paths for SAM2 imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'utils', 'sam2_api'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'utils', 'draw-euler'))

# Try to import SAM2 head detector
try:
    from sam2_head_detector import detect_head_with_sam2
except ImportError:
    print("Warning: Could not import sam2_head_detector, will use fallback face detection")
    detect_head_with_sam2 = None

# Import SAM2 API
try:
    from video_segmentation import SAM2VideoSegmenter, ClickPoint, SegmentationConfig
    SAM2_API_AVAILABLE = True
except ImportError:
    print("Warning: SAM2 API not available")
    SAM2_API_AVAILABLE = False


def detect_face_opencv(frame: np.ndarray) -> np.ndarray:
    """Fallback face detection using OpenCV Haar Cascade"""
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=4,
        minSize=(50, 50)
    )
    
    mask = np.zeros(gray.shape, dtype=np.uint8)
    for (x, y, w, h) in faces:
        # Add 20% margin
        margin = int(w * 0.2)
        x = max(0, x - margin)
        y = max(0, y - margin)
        w = w + (margin * 2)
        h = h + (margin * 2)
        # Draw filled rectangle for face region
        cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
    
    return mask


def detect_head_in_frame(frame: np.ndarray, frame_idx: int) -> np.ndarray:
    """
    Detect head/face in a single frame using SAM2 or fallback methods.
    Returns binary mask (255 for head, 0 for background).
    """
    
    # Try SAM2 head detector first
    if detect_head_with_sam2 is not None:
        try:
            print(f"  Frame {frame_idx}: Trying SAM2 head detection...")
            solid_mask, outline_mask, head_box = detect_head_with_sam2(
                frame, debug_name=f"frame_{frame_idx}"
            )
            
            if solid_mask is not None:
                print(f"  Frame {frame_idx}: SAM2 detection successful")
                return solid_mask
        except Exception as e:
            print(f"  Frame {frame_idx}: SAM2 failed: {e}")
    
    # Fallback to OpenCV face detection
    print(f"  Frame {frame_idx}: Using OpenCV face detection fallback")
    return detect_face_opencv(frame)


def create_head_detection_debug_video(input_video: str, output_path: str, sample_every: int = 10):
    """
    Create a debug video showing head/face detection masks.
    
    Args:
        input_video: Path to input video
        output_path: Output video path
        sample_every: Process every Nth frame (for speed)
    """
    
    # Open video
    cap = cv2.VideoCapture(input_video)
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video properties: {width}x{height}, {fps} fps, {total_frames} frames")
    print(f"Processing every {sample_every} frames for efficiency")
    
    # Create output writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width * 2, height))  # Side-by-side output
    
    print(f"\nProcessing frames for head detection...")
    
    # Cache for interpolating between sampled frames
    last_mask = None
    next_mask = None
    last_frame_idx = -1
    next_frame_idx = -1
    
    for frame_idx in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        
        # Get or interpolate mask
        if frame_idx % sample_every == 0:
            # Process this frame
            print(f"\nProcessing frame {frame_idx}/{total_frames}")
            mask = detect_head_in_frame(frame, frame_idx)
            
            # Update cache
            last_mask = mask
            last_frame_idx = frame_idx
            next_mask = None
            next_frame_idx = -1
            
            # Pre-compute next mask if possible
            if frame_idx + sample_every < total_frames:
                next_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx + sample_every)
                ret_next, next_frame = cap.read()
                if ret_next:
                    print(f"  Pre-computing frame {frame_idx + sample_every}")
                    next_mask = detect_head_in_frame(next_frame, frame_idx + sample_every)
                    next_frame_idx = frame_idx + sample_every
                cap.set(cv2.CAP_PROP_POS_FRAMES, next_pos)
                
        else:
            # Interpolate between last and next mask
            if last_mask is not None and next_mask is not None:
                # Linear interpolation
                alpha = (frame_idx - last_frame_idx) / (next_frame_idx - last_frame_idx)
                mask = cv2.addWeighted(last_mask, 1 - alpha, next_mask, alpha, 0)
                mask = (mask > 127).astype(np.uint8) * 255
            elif last_mask is not None:
                # Use last mask
                mask = last_mask
            else:
                # No mask available
                mask = np.zeros((height, width), dtype=np.uint8)
        
        # Create visualization
        # Convert mask to 3-channel
        mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        
        # Add colored overlay on original frame
        overlay = frame.copy()
        red_mask = np.zeros_like(frame)
        red_mask[:, :, 2] = mask  # Red channel
        overlay = cv2.addWeighted(overlay, 0.7, red_mask, 0.3, 0)
        
        # Add text labels
        cv2.putText(overlay, "Original + Head Overlay", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(mask_3ch, "Head Mask (Binary)", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Add statistics
        head_pixels = np.count_nonzero(mask)
        total_pixels = width * height
        coverage = (head_pixels / total_pixels) * 100
        
        stats_text = f"Frame {frame_idx} | Head: {head_pixels:,} px ({coverage:.1f}%)"
        cv2.putText(overlay, stats_text, (10, height - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(mask_3ch, stats_text, (10, height - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Combine side by side
        combined = np.hstack([overlay, mask_3ch])
        out.write(combined)
        
        if frame_idx % 30 == 0 and frame_idx > 0:
            print(f"  Progress: {frame_idx}/{total_frames} frames ({coverage:.1f}% head coverage)")
    
    # Clean up
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
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
    output_video = "sam2_head_detection_debug.mp4"
    
    # Process every 10th frame for speed (will interpolate between)
    final_video = create_head_detection_debug_video(
        input_video, 
        output_video,
        sample_every=10
    )
    
    print(f"\n✅ SAM2 head detection debug video created: {final_video}")
    print("\nVisualization shows:")
    print("  • Left: Original frame with red overlay on detected head")
    print("  • Right: Binary mask (white = head pixels)")
    print("  • Frame statistics and coverage percentage")
    print("  • Interpolation between sampled frames for smooth playback")


if __name__ == "__main__":
    main()