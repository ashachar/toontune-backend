#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Improved sandwich compositing that handles gray edge pixels.
"""

import cv2
import numpy as np
import subprocess
import os

def burn_ass_to_video(input_video: str, ass_file: str, temp_output: str):
    """Burn ASS subtitles into video using FFmpeg."""
    cmd = [
        "ffmpeg", "-y",
        "-i", input_video,
        "-vf", f"subtitles={ass_file}",
        "-c:v", "libx264",
        "-preset", "fast", 
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        temp_output
    ]
    
    print(f"Burning ASS subtitles to video...")
    subprocess.run(cmd, check=True, capture_output=True)
    print(f"Created video with burned subtitles: {temp_output}")

def detect_person_mask_improved(mask_frame):
    """
    Improved mask detection that handles gray edge pixels.
    Only marks gray pixels as background if they're adjacent to green screen.
    """
    # Green screen color and tolerance
    green_screen_color = np.array([154, 254, 119], dtype=np.uint8)
    green_tolerance = 20
    
    # Calculate difference from green screen color
    diff_from_green = np.abs(mask_frame.astype(np.int16) - green_screen_color.astype(np.int16))
    is_green = np.all(diff_from_green <= green_tolerance, axis=2)
    
    # Define the specific gray shade we're looking for (edge artifacts)
    # Based on analysis: gray edges are around BGR [109-112, 124-129, 119-127]
    target_gray = np.array([110, 125, 120], dtype=np.uint8)  # Average of found grays
    gray_tolerance = 10
    
    # Find pixels close to this specific gray shade
    diff_from_gray = np.abs(mask_frame.astype(np.int16) - target_gray.astype(np.int16))
    is_edge_gray = np.all(diff_from_gray <= gray_tolerance, axis=2)
    
    # Dilate the green mask to find pixels adjacent to green
    kernel = np.ones((3,3), np.uint8)
    green_dilated = cv2.dilate(is_green.astype(np.uint8), kernel, iterations=1)
    adjacent_to_green = (green_dilated == 1) & (~is_green)  # Adjacent but not green itself
    
    # Only mark gray pixels as background if they're adjacent to green
    gray_edge_artifacts = is_edge_gray & adjacent_to_green
    
    # Expand background to include green pixels AND gray edges adjacent to green
    background = is_green | gray_edge_artifacts
    
    # Foreground is everything that's NOT background
    person_mask = (~background).astype(np.uint8)
    
    # Apply light morphological operations to clean up
    kernel_small = np.ones((2,2), np.uint8)
    
    # Light opening to remove noise
    person_mask = cv2.morphologyEx(person_mask, cv2.MORPH_OPEN, kernel_small)
    
    # Light closing to fill small gaps
    person_mask = cv2.morphologyEx(person_mask, cv2.MORPH_CLOSE, kernel_small)
    
    return person_mask

def apply_foreground_sandwich(
    video_with_text: str,
    original_video: str, 
    mask_video: str,
    output_path: str
):
    """Apply foreground pixels on top of video that already has text burned in."""
    
    # Open all three videos
    cap_text = cv2.VideoCapture(video_with_text)
    cap_orig = cv2.VideoCapture(original_video)
    cap_mask = cv2.VideoCapture(mask_video)
    
    # Get video properties
    fps = cap_text.get(cv2.CAP_PROP_FPS)
    width = int(cap_text.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap_text.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap_text.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create output writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"Processing {total_frames} frames with improved edge handling...")
    
    for frame_idx in range(total_frames):
        ret_text, frame_with_text = cap_text.read()
        ret_orig, frame_original = cap_orig.read()
        ret_mask, mask_frame = cap_mask.read()
        
        if not ret_text or not ret_orig or not ret_mask:
            break
        
        # Get improved person mask
        person_mask = detect_person_mask_improved(mask_frame)
        
        # Convert to 3-channel
        person_mask_3ch = np.stack([person_mask, person_mask, person_mask], axis=2)
        
        # Hard composite: COMPLETELY replace with original where person is
        final_frame = np.where(person_mask_3ch == 1, frame_original, frame_with_text)
        final_frame = final_frame.astype(np.uint8)
        
        out.write(final_frame)
        
        if frame_idx % 30 == 0:
            print(f"Processed {frame_idx}/{total_frames} frames")
            
            # Debug: show mask statistics
            fg_pixels = np.sum(person_mask == 1)
            total_pixels = person_mask.size
            print(f"  Foreground: {100*fg_pixels/total_pixels:.1f}%")
    
    # Clean up
    cap_text.release()
    cap_orig.release()
    cap_mask.release()
    out.release()
    cv2.destroyAllWindows()
    
    print(f"Video saved: {output_path}")
    
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
    # Step 1: Burn ASS subtitles to video
    input_video = "ai_math1_6sec.mp4"
    ass_file = "ai_math1_wordbyword_captions.ass"
    temp_video_with_text = "temp_with_text.mp4"
    
    burn_ass_to_video(input_video, ass_file, temp_video_with_text)
    
    # Step 2: Apply foreground on top with improved edge handling
    mask_video = "../../uploads/assets/videos/ai_math1/ai_math1_rvm_mask.mp4"
    output_video = "ai_math1_improved_sandwich.mp4"
    
    final_video = apply_foreground_sandwich(
        temp_video_with_text,
        input_video,
        mask_video,
        output_video
    )
    
    # Clean up temp file
    os.remove(temp_video_with_text)
    
    print(f"\n✅ Final video with improved edge handling: {final_video}")

if __name__ == "__main__":
    main()