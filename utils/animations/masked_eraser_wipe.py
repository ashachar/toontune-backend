#!/usr/bin/env python3
"""
Masked eraser wipe - uses geometric masking instead of chromakey
This keeps the character fully opaque while creating the erase effect
FIXED: Uses Y-clamping to ensure eraser bottom is NEVER visible
"""

import subprocess
import os
import math


def create_masked_eraser_wipe(character_video: str, original_video: str, 
                              eraser_image: str, output_video: str,
                              wipe_start: float = 0, wipe_duration: float = 0.6):
    """
    Create eraser wipe using geometric masks instead of chromakey.
    CRITICAL FIX: Y-coordinate is clamped to ensure eraser bottom never enters frame.
    """
    
    print(f"Creating masked eraser wipe with Y-clamping fix...")
    
    # Get video properties
    probe_cmd = [
        'ffprobe', '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height,r_frame_rate',
        '-of', 'csv=s=x:p=0',
        character_video
    ]
    result = subprocess.run(probe_cmd, capture_output=True, text=True)
    parts = result.stdout.strip().split('x')
    width = int(parts[0])
    height = int(parts[1])
    
    # Define eraser path
    center_x = width // 2
    center_y = height // 2
    radius_x = 200
    radius_y = 150
    
    # Eraser positioning calculations
    # The eraser moves in an elliptical path
    # At highest point: y = center_y - radius_y * 0.8 = 180 - 120 = 60
    # At lowest point: y = center_y + radius_y * 0.8 = 180 + 120 = 300
    
    # Pre-composite character on black background (ensures full opacity)
    temp_composite = "outputs/temp_character_on_black.mp4"
    cmd_composite = [
        'ffmpeg', '-y',
        '-f', 'lavfi', '-i', f'color=black:s={width}x{height}',
        '-i', character_video,
        '-filter_complex', '[0:v][1:v]overlay=shortest=1',
        '-c:v', 'libx264',
        '-preset', 'fast',
        '-crf', '18',
        temp_composite
    ]
    
    print("Pre-compositing character on black background for full opacity...")
    result = subprocess.run(cmd_composite, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Failed to composite: {result.stderr}")
        # Use character video directly
        temp_composite = character_video
    
    # Create progressive reveal using crop and overlay (no chromakey!)
    # We'll reveal the original video in growing circles
    num_steps = 8
    
    filter_parts = []
    filter_parts.append(f"[0:v]setpts=PTS-STARTPTS[char];")  # Character video (opaque)
    filter_parts.append(f"[1:v]setpts=PTS-STARTPTS[orig];")  # Original video
    
    # CRITICAL: Keep eraser at full size with alpha preserved
    # Format as RGBA to maintain transparency, no scaling needed (already 1344px tall)
    filter_parts.append(f"[2:v]format=rgba,setsar=1[eraser];")  # Preserve full 768x1344 size
    
    # Build progressive reveal masks
    for i in range(num_steps):
        progress = i / (num_steps - 1) if num_steps > 1 else 0
        angle = 2 * math.pi * progress
        
        x = int(center_x + radius_x * math.cos(angle))
        y = int(center_y + radius_y * 0.8 * math.sin(angle))
        
        # Time when this reveal happens
        reveal_time = wipe_start + (progress * wipe_duration)
        
        # Create a circular reveal at this position
        # We'll use multiple overlays to progressively show more of the original
        if i == 0:
            # Start with character (no split needed for the fixed version)
            current = "char"
        
        # Crop a circle from the original and overlay it
        circle_size = 150
        crop_x = max(0, x - circle_size // 2)
        crop_y = max(0, y - circle_size // 2)
        
        filter_parts.append(
            f"[orig]crop={circle_size}:{circle_size}:{crop_x}:{crop_y}[crop{i}];"
            f"[{current}][crop{i}]overlay={crop_x}:{crop_y}:"
            f"enable='gte(t,{reveal_time})'[step{i}];"
        )
        current = f"step{i}"
    
    # Add the moving eraser on top with Y-CLAMPING FIX
    # CRITICAL: Use max() to clamp Y so eraser bottom NEVER enters frame
    # The eraser is 1344px tall, frame is 360px tall
    # We want to show about 400-500px of the hand/arm in frame
    
    # X-position: centered with offset (150/768 ≈ 0.1953 of width for hand alignment)
    pivot_x_ratio = 0.1953
    
    # Y-position: elliptical motion but CLAMPED to keep bottom off-screen
    # Raw elliptical Y (without clamp)
    y_raw = f"{center_y}+{radius_y*0.8}*sin(2*PI*(t-{wipe_start})/{wipe_duration})"
    
    # Clamp: We want to show about 450px of the hand/arm in the frame
    # So the eraser top should never go below: 360 - 450 = -90
    # This ensures bottom (at -90 + 1344 = 1254) is way past frame bottom (360)
    # But we still see a good portion of the hand
    visible_hand_height = 450  # How much of hand/arm we want visible
    min_y = f"main_h - {visible_hand_height}"  # 360 - 450 = -90
    
    eraser_motion = (f"overlay="
                    f"x='{center_x}+{radius_x}*cos(2*PI*(t-{wipe_start})/{wipe_duration})-overlay_w*{pivot_x_ratio}':"
                    f"y='max({y_raw},{min_y})':"  # Clamp to show hand but keep bottom off-screen
                    f"eval=frame:"  # Per-frame evaluation for accurate motion
                    f"enable='between(t,{wipe_start},{wipe_start + wipe_duration + 0.02})'")  # Slight extension to avoid end-frame pop
    
    filter_parts.append(f"[{current}][eraser]{eraser_motion}[with_eraser];")
    # Ensure output stays at video dimensions
    filter_parts.append(f"[with_eraser]format=yuv420p")
    
    # Join all filter parts
    filter_complex = "".join(filter_parts)
    
    cmd = [
        'ffmpeg', '-y',
        '-i', temp_composite,  # Character (fully opaque)
        '-i', original_video,  # Original to reveal
        '-i', eraser_image,    # Eraser
        '-filter_complex', filter_complex,
        '-c:v', 'libx264',
        '-preset', 'fast',
        '-crf', '18',
        '-pix_fmt', 'yuv420p',
        output_video
    ]
    
    print(f"Applying masked reveal (no transparency)...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Clean up temp file
    if os.path.exists(temp_composite) and temp_composite != character_video:
        os.remove(temp_composite)
    
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        
        # Simpler fallback: just blend at 50% point
        print("Trying simpler approach...")
        
        cmd_simple = [
            'ffmpeg', '-y',
            '-i', character_video,
            '-i', original_video,
            '-i', eraser_image,
            '-filter_complex',
            f"[0:v][1:v]overlay=enable='gte(t,{wipe_start + wipe_duration/2})'[merged];"
            f"[2:v]scale=300:-1[eraser];"
            f"[merged][eraser]overlay="
            f"x='{center_x}+{radius_x}*cos(2*PI*(t-{wipe_start})/{wipe_duration})-150':"
            f"y='{center_y}+{radius_y*0.8}*sin(2*PI*(t-{wipe_start})/{wipe_duration})-100':"
            f"enable='between(t,{wipe_start},{wipe_start + wipe_duration})'",
            '-c:v', 'libx264',
            '-preset', 'fast',
            '-crf', '18',
            '-pix_fmt', 'yuv420p',
            output_video
        ]
        
        result = subprocess.run(cmd_simple, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"Simple approach also failed: {result.stderr}")
            return False
    
    print(f"Masked eraser wipe created: {output_video}")
    return True