#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple sandwich compositing: ASS -> burn to video -> paste foreground on top.
Much simpler and uses existing ASS rendering!
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
        "-crf", "18",  # High quality
        "-pix_fmt", "yuv420p",
        temp_output
    ]
    
    print(f"Burning ASS subtitles to video...")
    subprocess.run(cmd, check=True, capture_output=True)
    print(f"Created video with burned subtitles: {temp_output}")

def apply_foreground_sandwich(
    video_with_text: str,
    original_video: str, 
    mask_video: str,
    output_path: str
):
    """Apply foreground pixels on top of video that already has text burned in."""
    
    # Open all three videos
    cap_text = cv2.VideoCapture(video_with_text)  # Video with ASS text burned in
    cap_orig = cv2.VideoCapture(original_video)   # Original clean video
    cap_mask = cv2.VideoCapture(mask_video)       # Mask video
    
    # Get video properties
    fps = cap_text.get(cv2.CAP_PROP_FPS)
    width = int(cap_text.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap_text.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap_text.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create output writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"Processing {total_frames} frames...")
    
    for frame_idx in range(total_frames):
        ret_text, frame_with_text = cap_text.read()
        ret_orig, frame_original = cap_orig.read()
        ret_mask, mask_frame = cap_mask.read()
        
        if not ret_text or not ret_orig or not ret_mask:
            break
        
        # The mask uses green screen: BGR [154, 254, 119] for background
        # Everything else is foreground (person)
        green_screen_color = np.array([154, 254, 119], dtype=np.uint8)
        
        # Use tolerance to handle color variation from Replicate
        tolerance = 20  # Increased tolerance for better coverage
        
        # Calculate difference from green screen color
        diff = np.abs(mask_frame.astype(np.int16) - green_screen_color.astype(np.int16))
        
        # Check if within tolerance for all channels
        is_green_screen = np.all(diff <= tolerance, axis=2)
        
        # Foreground mask: 1 where person is (NOT green), 0 where background (green)
        person_mask = (~is_green_screen).astype(np.uint8)
        
        # Apply morphological operations to clean edges
        kernel = np.ones((3,3), np.uint8)
        person_mask = cv2.morphologyEx(person_mask, cv2.MORPH_OPEN, kernel)
        person_mask = cv2.morphologyEx(person_mask, cv2.MORPH_CLOSE, kernel)
        
        # Convert to 3-channel
        person_mask_3ch = np.stack([person_mask, person_mask, person_mask], axis=2)
        
        # Hard composite: COMPLETELY replace with original where person is
        # No blending - just binary selection based on green screen
        final_frame = np.where(person_mask_3ch == 1, frame_original, frame_with_text)
        final_frame = final_frame.astype(np.uint8)
        
        out.write(final_frame)
        
        if frame_idx % 30 == 0:
            print(f"Processed {frame_idx}/{total_frames} frames")
    
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
    
    # Step 2: Apply foreground on top
    mask_video = "../../uploads/assets/videos/ai_math1/ai_math1_rvm_mask.mp4"
    output_video = "ai_math1_simple_sandwich.mp4"
    
    final_video = apply_foreground_sandwich(
        temp_video_with_text,
        input_video,
        mask_video,
        output_video
    )
    
    # Clean up temp file
    os.remove(temp_video_with_text)
    
    print(f"\nFinal video with text behind head: {final_video}")

if __name__ == "__main__":
    main()