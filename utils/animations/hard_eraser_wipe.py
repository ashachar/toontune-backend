#!/usr/bin/env python3
"""
Hard eraser wipe animation - abruptly deletes pixels near eraser
"""

import subprocess
import os
import numpy as np
import cv2
import tempfile


def create_hard_eraser_wipe(character_video: str, original_video: str, 
                            eraser_image: str, output_video: str,
                            wipe_start: float = 2.7, wipe_duration: float = 0.6):
    """
    Create eraser wipe that abruptly deletes pixels within radius of eraser.
    Uses a growing mask approach with FFmpeg.
    """
    
    # First, get video properties
    cap = cv2.VideoCapture(character_video)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    # Calculate frame range
    wipe_start_frame = int(wipe_start * fps)
    wipe_end_frame = int((wipe_start + wipe_duration) * fps)
    
    # Create a series of mask overlays for each position
    # We'll use drawbox filter to create growing erased areas
    
    # Define eraser path (circular motion around character)
    center_x = width // 2
    center_y = height // 2
    radius_x = 200
    radius_y = 150
    
    # Number of steps in the eraser path
    num_steps = 12
    erase_radius = 80  # Radius of pixels to erase around eraser
    
    # Build complex filter chain
    filter_parts = []
    
    # Create cumulative masking - each step adds to the previous mask
    for i in range(num_steps):
        t = i / num_steps
        angle = 2 * np.pi * t
        
        # Eraser position
        x = int(center_x + radius_x * np.cos(angle))
        y = int(center_y + radius_y * np.sin(angle) * 0.8)
        
        # Time when this erasure happens
        step_time = wipe_start + (i * wipe_duration / num_steps)
        
        # Create a drawbox that acts as a mask
        # We'll use multiple overlapping boxes to approximate a circle
        box_filters = []
        
        # Create a rough circle using overlapping rectangles
        for angle_offset in [0, 45, 90, 135]:
            offset_rad = np.radians(angle_offset)
            box_w = int(erase_radius * 1.5)
            box_h = int(erase_radius * 1.5)
            box_x = x - box_w // 2 + int(20 * np.cos(offset_rad))
            box_y = y - box_h // 2 + int(20 * np.sin(offset_rad))
            
            # Ensure coordinates are within frame
            box_x = max(0, min(box_x, width - box_w))
            box_y = max(0, min(box_y, height - box_h))
            
            box_filters.append(f"drawbox=x={box_x}:y={box_y}:w={box_w}:h={box_h}:"
                             f"color=green@1.0:thickness=fill:"
                             f"enable='gte(t,{step_time})'")
    
    # Build the filter complex
    # Strategy: Draw green boxes on character video where eraser has been,
    # then use chromakey to remove those areas and show original video
    
    filter_chain = "[0:v]"
    
    # Add all the drawbox filters
    for i in range(num_steps):
        t = i / num_steps
        angle = 2 * np.pi * t
        x = int(center_x + radius_x * np.cos(angle))
        y = int(center_y + radius_y * np.sin(angle) * 0.8)
        step_time = wipe_start + (i * wipe_duration / num_steps)
        
        # Draw a filled circle (approximated with box) at eraser position
        box_size = erase_radius * 2
        box_x = max(0, x - box_size // 2)
        box_y = max(0, y - box_size // 2)
        
        filter_chain += f"drawbox=x={box_x}:y={box_y}:w={box_size}:h={box_size}:"
        filter_chain += f"color=black@1.0:thickness=fill:"
        filter_chain += f"enable='gte(t,{step_time})',"
    
    # Remove trailing comma and add output label
    filter_chain = filter_chain.rstrip(',') + "[masked];"
    
    # Add eraser overlay animation
    eraser_motion = []
    for i in range(num_steps):
        t = i / num_steps
        t_next = (i + 1) / num_steps if i < num_steps - 1 else 1.0
        angle = 2 * np.pi * t
        
        x = int(center_x + radius_x * np.cos(angle)) - 150  # Offset for eraser image
        y = int(center_y + radius_y * np.sin(angle) * 0.8) - 100
        
        step_start = wipe_start + (i * wipe_duration / num_steps)
        step_end = wipe_start + (t_next * wipe_duration)
        
        eraser_motion.append(f"overlay=x={x}:y={y}:enable='between(t,{step_start},{step_end})'")
    
    # Complete filter: mask character video, then overlay with original where masked
    complete_filter = (
        f"[2:v]scale=300:-1[eraser];"  # Scale eraser
        f"{filter_chain}"  # Apply masking boxes
        f"[masked][1:v]overlay=enable='gte(t,{wipe_start})'[with_original];"  # Show original where masked
        f"[with_original][eraser]{eraser_motion[0]}[with_eraser];"  # Add moving eraser
    )
    
    # Add remaining eraser positions
    for i in range(1, min(len(eraser_motion), 3)):  # Limit to avoid filter complexity
        complete_filter += f"[with_eraser][eraser]{eraser_motion[i]}[with_eraser{i}];"
        if i == min(len(eraser_motion), 3) - 1:
            complete_filter = complete_filter.replace(f"[with_eraser{i}]", "")  # Final output
    
    # Simplified approach using blend with hard mask transitions
    # Create a mask video that grows as eraser moves
    
    # Even simpler: use multiple crop and overlay filters
    
    # Simpler approach: Create growing erased area along the path
    # Check multiple points along the path up to current progress
    num_checks = 12  # Number of points to check along the path
    
    # Build conditions for cumulative erasing
    conditions = []
    for i in range(num_checks):
        progress = i / (num_checks - 1) if num_checks > 1 else 0
        angle = 2 * np.pi * progress
        
        # Eraser position at this progress point
        x = int(center_x + radius_x * np.cos(angle))
        y = int(center_y + radius_y * np.sin(angle) * 0.8)
        
        # Time when eraser reaches this position
        time_at_pos = wipe_start + (progress * wipe_duration)
        
        # If we've passed this time AND pixel is within radius, show B
        condition = f"(gte(T,{time_at_pos})*lt(hypot(X-{x},Y-{y}),{erase_radius}))"
        conditions.append(condition)
    
    # Join all conditions with OR logic (sum > 0 means at least one is true)
    cumulative_expr = "+".join(conditions)
    
    simple_filter = f"""
    [2:v]scale=300:-1[eraser];
    [0:v][1:v]blend=all_expr='if(lt(T,{wipe_start}),A,if(gt(T,{wipe_start + wipe_duration}),B,if({cumulative_expr},B,A)))'[blended];
    [blended][eraser]overlay=x='{center_x}+{radius_x}*cos(2*PI*(t-{wipe_start})/{wipe_duration})-150':y='{center_y}+{int(radius_y*0.8)}*sin(2*PI*(t-{wipe_start})/{wipe_duration})-100':enable='between(t,{wipe_start},{wipe_start + wipe_duration})'
    """
    
    cmd = [
        'ffmpeg', '-y',
        '-i', character_video,  # Character video (to be erased)
        '-i', original_video,   # Original video (to be revealed)  
        '-i', eraser_image,     # Eraser image
        '-filter_complex', simple_filter,
        '-c:v', 'libx264',
        '-preset', 'fast',
        '-crf', '18',
        '-pix_fmt', 'yuv420p',
        output_video
    ]
    
    print(f"Creating hard eraser wipe...")
    print(f"Erase radius: {erase_radius} pixels")
    print(f"Animation: {wipe_start}s to {wipe_start + wipe_duration}s")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        
        # Fallback to simpler approach
        print("Trying simpler approach...")
        
        # Very simple: just reveal original in steps
        simple_filter2 = f"""
        [2:v]scale=300:-1[eraser];
        [0:v][1:v]overlay=enable='gte(t,{wipe_start + wipe_duration*0.5})'[revealed];
        [revealed][eraser]overlay=x='{center_x}+{radius_x}*cos(2*PI*(t-{wipe_start})/{wipe_duration})-150':y='{center_y}+{int(radius_y*0.8)}*sin(2*PI*(t-{wipe_start})/{wipe_duration})-100':enable='between(t,{wipe_start},{wipe_start + wipe_duration})'
        """
        
        cmd[6] = simple_filter2
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"Fallback also failed: {result.stderr}")
            return False
    
    print(f"Hard eraser wipe created: {output_video}")
    return True