#!/usr/bin/env python3
"""
Progressive eraser wipe - uses FFmpeg drawbox to create cumulative erasing
"""

import subprocess
import os


def create_progressive_eraser_wipe(character_video: str, original_video: str, 
                                   eraser_image: str, output_video: str,
                                   wipe_start: float = 2.7, wipe_duration: float = 0.6):
    """
    Create eraser wipe where pixels progressively get erased and stay erased.
    Uses FFmpeg drawbox filters to create cumulative reveal areas.
    """
    
    # Get video dimensions
    probe_cmd = [
        'ffprobe', '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height',
        '-of', 'csv=s=x:p=0',
        character_video
    ]
    result = subprocess.run(probe_cmd, capture_output=True, text=True)
    width, height = map(int, result.stdout.strip().split('x'))
    
    # Define eraser path
    center_x = width // 2
    center_y = height // 2
    radius_x = 200
    radius_y = 150
    erase_size = 150  # Size of erase box
    
    # Build progressive drawbox filters
    # Each box appears when eraser reaches that position and stays
    num_boxes = 8  # Number of boxes along the path
    
    drawbox_filters = []
    for i in range(num_boxes):
        progress = i / (num_boxes - 1) if num_boxes > 1 else 0
        angle = 2 * 3.14159 * progress
        
        # Position for this box
        import math
        x = int(center_x + radius_x * math.cos(angle) - erase_size/2)
        y = int(center_y + radius_y * 0.8 * math.sin(angle) - erase_size/2)
        
        # Time when this box appears
        box_time = wipe_start + (progress * wipe_duration)
        
        # Create green box that appears at box_time and stays
        # Green will be keyed out to reveal original video
        box_filter = f"drawbox=x={x}:y={y}:w={erase_size}:h={erase_size}:color=green@1.0:thickness=fill:enable='gte(t,{box_time})'"
        drawbox_filters.append(box_filter)
    
    # Join all drawbox filters
    drawbox_chain = ",".join(drawbox_filters)
    
    # Build complete filter
    filter_complex = f"""
    [3:v]scale=300:-1[eraser];
    [0:v]{drawbox_chain}[marked];
    [marked]chromakey=green:0.15:0.1[keyed];
    [1:v][keyed]overlay[merged];
    [merged][eraser]overlay=
        x='{center_x}+{radius_x}*cos(2*PI*(t-{wipe_start})/{wipe_duration})-150':
        y='{center_y}+{radius_y*0.8}*sin(2*PI*(t-{wipe_start})/{wipe_duration})-100':
        enable='between(t,{wipe_start},{wipe_start + wipe_duration})'
    """
    
    cmd = [
        'ffmpeg', '-y',
        '-i', character_video,  # Character video
        '-i', original_video,   # Original video  
        '-stream_loop', '-1',   # Loop original video if needed
        '-t', str(wipe_start + wipe_duration + 1),  # Duration
        '-i', original_video,   # Original video again for proper duration
        '-i', eraser_image,     # Eraser image
        '-filter_complex', filter_complex,
        '-c:v', 'libx264',
        '-preset', 'fast',
        '-crf', '18',
        '-pix_fmt', 'yuv420p',
        output_video
    ]
    
    print(f"Creating progressive eraser wipe...")
    print(f"Animation: {wipe_start}s to {wipe_start + wipe_duration}s")
    print(f"Creating {num_boxes} progressive erase zones")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        
        # Simpler fallback: just two-stage reveal
        print("Trying simpler two-stage approach...")
        
        filter_simple = f"""
        [2:v]scale=300:-1[eraser];
        [0:v]drawbox=x=0:y=0:w='iw':h='ih':color=green@1.0:thickness=fill:enable='gte(t,{wipe_start + wipe_duration*0.5})'[marked];
        [marked]chromakey=green:0.15:0.1[keyed];
        [1:v][keyed]overlay[merged];
        [merged][eraser]overlay=
            x='{center_x}+{radius_x}*cos(2*PI*(t-{wipe_start})/{wipe_duration})-150':
            y='{center_y}+{radius_y*0.8}*sin(2*PI*(t-{wipe_start})/{wipe_duration})-100':
            enable='between(t,{wipe_start},{wipe_start + wipe_duration})'
        """
        
        cmd_simple = [
            'ffmpeg', '-y',
            '-i', character_video,
            '-i', original_video,
            '-i', eraser_image,
            '-filter_complex', filter_simple,
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
    
    print(f"Progressive eraser wipe created: {output_video}")
    return True