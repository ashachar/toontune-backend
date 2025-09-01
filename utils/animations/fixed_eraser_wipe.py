#!/usr/bin/env python3
"""
Fixed eraser wipe - ensures eraser bottom is never visible
Key fix: Properly handle overlay positioning with tall images
"""

import subprocess
import os
import math


def create_masked_eraser_wipe(character_video: str, original_video: str, 
                              eraser_image: str, output_video: str,
                              wipe_start: float = 0, wipe_duration: float = 0.6):
    """
    Create eraser wipe with eraser bottom always off-screen.
    FIXED: Ensures proper overlay dimensions and positioning.
    """
    
    print(f"Creating fixed eraser wipe...")
    
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
    width = int(parts[0])  # 640
    height = int(parts[1])  # 360
    
    print(f"Video dimensions: {width}x{height}")
    
    # Get eraser dimensions
    probe_eraser = [
        'ffprobe', '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height',
        '-of', 'csv=s=x:p=0',
        eraser_image
    ]
    result = subprocess.run(probe_eraser, capture_output=True, text=True)
    eraser_parts = result.stdout.strip().split('x')
    eraser_width = int(eraser_parts[0])  # 768
    eraser_height = int(eraser_parts[1])  # 1344
    
    print(f"Original eraser dimensions: {eraser_width}x{eraser_height}")
    
    # Define eraser path
    center_x = width // 2  # 320
    center_y = height // 2  # 180
    radius_x = 200
    radius_y = 150
    
    # Calculate the highest point the eraser will reach
    highest_y = center_y - int(radius_y * 0.8)  # 180 - 120 = 60
    
    # Calculate required eraser height to ensure bottom is always off-screen
    # The eraser needs to extend from highest_y to beyond the frame bottom
    # Add extra margin to be absolutely sure
    required_extension = height - highest_y + 100  # 360 - 60 + 100 = 400
    
    # Scale eraser to be tall enough
    # We want the eraser to be AT LEAST this tall
    min_eraser_height = required_extension + 200  # 600 minimum
    
    # Since original is 1344px, we should keep it or make it taller
    # Calculate scale factor
    if eraser_height < min_eraser_height:
        scale_factor = min_eraser_height / eraser_height
        scaled_height = min_eraser_height
        scaled_width = int(eraser_width * scale_factor)
    else:
        # Keep original size since it's already tall enough
        scaled_height = eraser_height
        scaled_width = eraser_width
    
    print(f"Scaled eraser will be: {scaled_width}x{scaled_height}")
    
    # Pre-composite character on black background
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
    
    print("Pre-compositing character on black background...")
    result = subprocess.run(cmd_composite, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Failed to composite: {result.stderr}")
        temp_composite = character_video
    
    # Build the filter complex
    # CRITICAL: We must ensure the output stays at original video dimensions
    filter_parts = []
    
    # Input streams
    filter_parts.append(f"[0:v]setpts=PTS-STARTPTS[char];")  # Character video
    filter_parts.append(f"[1:v]setpts=PTS-STARTPTS[orig];")  # Original video
    
    # Scale eraser to exact dimensions we calculated
    filter_parts.append(f"[2:v]scale={scaled_width}:{scaled_height}[eraser_scaled];")
    
    # Create progressive reveal using geometric masks
    num_steps = 8
    current = "char"
    
    for i in range(num_steps):
        progress = i / (num_steps - 1) if num_steps > 1 else 0
        angle = 2 * math.pi * progress
        
        x = int(center_x + radius_x * math.cos(angle))
        y = int(center_y + radius_y * 0.8 * math.sin(angle))
        
        reveal_time = wipe_start + (progress * wipe_duration)
        
        # Create circular reveal
        circle_size = 150
        crop_x = max(0, min(x - circle_size // 2, width - circle_size))
        crop_y = max(0, min(y - circle_size // 2, height - circle_size))
        
        filter_parts.append(
            f"[orig]crop={circle_size}:{circle_size}:{crop_x}:{crop_y}[crop{i}];"
            f"[{current}][crop{i}]overlay={crop_x}:{crop_y}:"
            f"enable='gte(t,{reveal_time})'[step{i}];"
        )
        current = f"step{i}"
    
    # Add the moving eraser
    # CRITICAL: Calculate Y position so eraser bottom is always below frame
    # The eraser top should be positioned so that even at highest point,
    # the bottom extends well past the frame
    
    # At highest point, we want significant portion of eraser visible
    # but bottom must be off-screen
    # Position calculation: ensure at least 300px of eraser is below frame bottom
    eraser_x_offset = scaled_width // 2  # Center the eraser horizontally
    
    # Y positioning: we want the eraser to move but always extend past bottom
    # At highest point (y=60), position eraser so its bottom is at least at y=500
    # This means eraser top should be at: 500 - scaled_height
    base_y_offset = -scaled_height + height + 200  # Ensure 200px extends past bottom
    
    eraser_motion = (
        f"overlay="
        f"x='{center_x}+{radius_x}*cos(2*PI*(t-{wipe_start})/{wipe_duration})-{eraser_x_offset}':"
        f"y='{center_y}+{radius_y*0.8}*sin(2*PI*(t-{wipe_start})/{wipe_duration})+{base_y_offset}':"
        f"enable='between(t,{wipe_start},{wipe_start + wipe_duration})'"
    )
    
    filter_parts.append(f"[{current}][eraser_scaled]{eraser_motion}[with_eraser];")
    
    # CRITICAL: Ensure output is cropped to original video dimensions
    filter_parts.append(f"[with_eraser]crop={width}:{height}:0:0")
    
    filter_complex = "".join(filter_parts)
    
    cmd = [
        'ffmpeg', '-y',
        '-i', temp_composite,  # Character
        '-i', original_video,  # Original
        '-i', eraser_image,    # Eraser
        '-filter_complex', filter_complex,
        '-c:v', 'libx264',
        '-preset', 'fast',
        '-crf', '18',
        '-pix_fmt', 'yuv420p',
        output_video
    ]
    
    print(f"Applying fixed eraser wipe...")
    print(f"Eraser will move with top ranging from y={center_y - int(radius_y*0.8) + base_y_offset} to y={center_y + int(radius_y*0.8) + base_y_offset}")
    print(f"Eraser bottom will always be at least at y={center_y + int(radius_y*0.8) + base_y_offset + scaled_height}")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Clean up temp file
    if os.path.exists(temp_composite) and temp_composite != character_video:
        os.remove(temp_composite)
    
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        return False
    
    print(f"Fixed eraser wipe created: {output_video}")
    return True