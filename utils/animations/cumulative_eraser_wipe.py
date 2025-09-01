#!/usr/bin/env python3
"""
Cumulative eraser wipe animation - pixels stay erased after being revealed
"""

import subprocess
import os
import numpy as np
import cv2
import tempfile


def create_cumulative_eraser_wipe(character_video: str, original_video: str, 
                                  eraser_image: str, output_video: str,
                                  wipe_start: float = 2.7, wipe_duration: float = 0.6):
    """
    Create eraser wipe where erased pixels stay visible permanently.
    Uses a growing mask approach where each eraser position adds to the reveal.
    """
    
    # First, get video properties
    cap = cv2.VideoCapture(character_video)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    print(f"Video properties: {width}x{height} @ {fps}fps, {total_frames} frames")
    
    # Calculate frame range
    wipe_start_frame = int(wipe_start * fps)
    wipe_end_frame = int((wipe_start + wipe_duration) * fps)
    
    # Define eraser path (circular motion around character)
    center_x = width // 2
    center_y = height // 2
    radius_x = 200
    radius_y = 150
    erase_radius = 100  # Radius of pixels to erase around eraser
    
    # Create a mask video that grows as eraser moves
    # This will be a green screen video where green areas show original
    mask_video = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False).name
    
    # Create mask frames more efficiently - only process wipe duration
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    mask_writer = cv2.VideoWriter(mask_video, fourcc, fps, (width, height))
    
    # Create cumulative mask (starts all black, grows green where erased)
    cumulative_mask = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Process only relevant frames for efficiency
    wipe_frames = wipe_end_frame - wipe_start_frame
    
    print(f"Generating mask for frames {wipe_start_frame} to {wipe_end_frame} ({wipe_frames} frames)")
    
    for frame_num in range(total_frames):
        if frame_num < wipe_start_frame:
            # Before wipe: all black (show character)
            mask_frame = np.zeros((height, width, 3), dtype=np.uint8)
        elif frame_num >= wipe_end_frame:
            # After wipe: complete green circle path
            # Draw the complete path instead of keeping cumulative
            for i in range(12):  # Draw 12 points along the path
                progress = i / 11
                angle = 2 * np.pi * progress
                x = int(center_x + radius_x * np.cos(angle))
                y = int(center_y + radius_y * np.sin(angle) * 0.8)
                cv2.circle(cumulative_mask, (x, y), erase_radius, (0, 255, 0), -1)
            mask_frame = cumulative_mask.copy()
        else:
            # During wipe: add current eraser position to cumulative mask
            progress = (frame_num - wipe_start_frame) / max(1, wipe_end_frame - wipe_start_frame)
            angle = 2 * np.pi * progress
            
            # Eraser position
            x = int(center_x + radius_x * np.cos(angle))
            y = int(center_y + radius_y * np.sin(angle) * 0.8)
            
            # Add circle at eraser position to cumulative mask (green = reveal original)
            cv2.circle(cumulative_mask, (x, y), erase_radius, (0, 255, 0), -1)
            mask_frame = cumulative_mask.copy()
        
        mask_writer.write(mask_frame)
        
        if frame_num % 30 == 0:
            print(f"  Processed frame {frame_num}/{total_frames}")
    
    mask_writer.release()
    print(f"Created cumulative mask video: {mask_video}")
    
    # Now use FFmpeg to apply the mask with eraser overlay
    cmd = [
        'ffmpeg', '-y',
        '-i', character_video,  # Character video (input 0)
        '-i', original_video,   # Original video (input 1)
        '-i', mask_video,       # Mask video (input 2)
        '-i', eraser_image,     # Eraser image (input 3)
        '-filter_complex',
        f"""
        [3:v]scale=300:-1[eraser];
        [2:v]format=yuv420p[mask];
        [0:v][1:v][mask]maskedmerge[merged];
        [merged][eraser]overlay=
            x='{center_x}+{radius_x}*cos(2*PI*(t-{wipe_start})/{wipe_duration})-150':
            y='{center_y}+{radius_y*0.8}*sin(2*PI*(t-{wipe_start})/{wipe_duration})-100':
            enable='between(t,{wipe_start},{wipe_start + wipe_duration})'
        """,
        '-c:v', 'libx264',
        '-preset', 'fast',
        '-crf', '18',
        '-pix_fmt', 'yuv420p',
        output_video
    ]
    
    print(f"Creating cumulative eraser wipe...")
    print(f"Erase radius: {erase_radius} pixels")
    print(f"Animation: {wipe_start}s to {wipe_start + wipe_duration}s")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Clean up temp mask video
    if os.path.exists(mask_video):
        os.remove(mask_video)
    
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        
        # Fallback: simpler approach with chromakey
        print("Trying chromakey approach...")
        
        # Create green screen mask video  
        mask_video2 = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False).name
        mask_writer2 = cv2.VideoWriter(mask_video2, fourcc, fps, (width, height))
        
        # Reset cumulative mask
        cumulative_mask = np.zeros((height, width, 3), dtype=np.uint8)
        
        for frame_num in range(total_frames):
            if frame_num < wipe_start_frame:
                # Before wipe: no green (show character)
                mask_frame = np.zeros((height, width, 3), dtype=np.uint8)
            elif frame_num >= wipe_end_frame:
                # After wipe: all green (show original)
                mask_frame = np.full((height, width, 3), (0, 255, 0), dtype=np.uint8)
            else:
                # During wipe: growing green area
                progress = (frame_num - wipe_start_frame) / max(1, wipe_end_frame - wipe_start_frame)
                angle = 2 * np.pi * progress
                
                x = int(center_x + radius_x * np.cos(angle))
                y = int(center_y + radius_y * np.sin(angle) * 0.8)
                
                # Add green circle to cumulative mask
                cv2.circle(cumulative_mask, (x, y), erase_radius, (0, 255, 0), -1)
                mask_frame = cumulative_mask.copy()
            
            mask_writer2.write(mask_frame)
        
        mask_writer2.release()
        
        # Apply using chromakey
        cmd2 = [
            'ffmpeg', '-y',
            '-i', character_video,
            '-i', original_video,
            '-i', mask_video2,
            '-i', eraser_image,
            '-filter_complex',
            f"""
            [3:v]scale=300:-1[eraser];
            [0:v][2:v]overlay[masked_char];
            [masked_char]chromakey=green:0.01:0.0[keyed];
            [1:v][keyed]overlay[with_original];
            [with_original][eraser]overlay=
                x='{center_x}+{radius_x}*cos(2*PI*(t-{wipe_start})/{wipe_duration})-150':
                y='{center_y}+{radius_y*0.8}*sin(2*PI*(t-{wipe_start})/{wipe_duration})-100':
                enable='between(t,{wipe_start},{wipe_start + wipe_duration})'
            """,
            '-c:v', 'libx264',
            '-preset', 'fast',
            '-crf', '18',
            '-pix_fmt', 'yuv420p',
            output_video
        ]
        
        result = subprocess.run(cmd2, capture_output=True, text=True)
        
        # Clean up
        if os.path.exists(mask_video2):
            os.remove(mask_video2)
        
        if result.returncode != 0:
            print(f"Chromakey approach also failed: {result.stderr}")
            return False
    
    print(f"Cumulative eraser wipe created: {output_video}")
    return True